// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_longrope_kv.hpp"

#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"
#include "util.hpp"
#include "util_xarch.hpp"

namespace {

// Turns one rotate_half key row: components j and j + half belong to the same plane.
template <typename T>
void rerotate_row(T* row, const float* delta_cos, const float* delta_sin, size_t half) {
    for (size_t j = 0; j < half; ++j) {
        const float a = static_cast<float>(row[j]);
        const float b = static_cast<float>(row[j + half]);
        row[j] = static_cast<T>(a * delta_cos[j] - b * delta_sin[j]);
        row[j + half] = static_cast<T>(b * delta_cos[j] + a * delta_sin[j]);
    }
}

template <typename T>
void rerotate_planes(T* data,
                     size_t outer,
                     size_t seq_len,
                     size_t seq_stride,
                     size_t rows_per_token,
                     size_t head_dim,
                     size_t num_tokens,
                     const ov::npuw::longrope::RegimeDelta& delta) {
    for (size_t o = 0; o < outer; ++o) {
        T* plane = data + o * seq_len * seq_stride;
        for (size_t t = 0; t < num_tokens; ++t) {
            const float* dcos = delta.cos.data() + t * delta.half;
            const float* dsin = delta.sin.data() + t * delta.half;
            T* token = plane + t * seq_stride;
            for (size_t r = 0; r < rows_per_token; ++r) {
                rerotate_row(token + r * head_dim, dcos, dsin, delta.half);
            }
        }
    }
}

}  // anonymous namespace

ov::npuw::longrope::RegimeDelta ov::npuw::longrope::make_regime_delta(
    ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
    int64_t first_position_id,
    uint32_t num_tokens,
    bool to_long) {
    RegimeDelta delta;
    if (num_tokens == 0u || !tables.has_long) {
        // Without a distinct long-factor set both regimes share the same coefficients,
        // so a flip changes nothing about the cached keys.
        return delta;
    }
    OPENVINO_ASSERT(tables.is_valid(), "NPUW: LongRoPE regime flipped but the cos/sin tables are missing.");

    const size_t rotary_ndims = tables.rotary_ndims;
    OPENVINO_ASSERT(rotary_ndims % 2u == 0u, "NPUW: LongRoPE rotary dimension must be even.");
    OPENVINO_ASSERT(first_position_id >= 0 && static_cast<size_t>(first_position_id) + num_tokens <= tables.max_len,
                    "NPUW: cached LongRoPE positions [",
                    first_position_id,
                    ", ",
                    first_position_id + num_tokens,
                    ") fall outside the coefficient tables (",
                    tables.max_len,
                    " rows).");

    delta.half = rotary_ndims / 2u;
    delta.cos.resize(static_cast<size_t>(num_tokens) * delta.half);
    delta.sin.resize(static_cast<size_t>(num_tokens) * delta.half);

    // Row p of either regime holds the coefficients of absolute position p.
    auto cos_old = tables.cos_rows(tables.max_len, !to_long);
    auto sin_old = tables.sin_rows(tables.max_len, !to_long);
    auto cos_new = tables.cos_rows(tables.max_len, to_long);
    auto sin_new = tables.sin_rows(tables.max_len, to_long);

    const size_t skip = static_cast<size_t>(first_position_id) * rotary_ndims;
    const auto* co = cos_old.data<ov::float16>() + skip;
    const auto* so = sin_old.data<ov::float16>() + skip;
    const auto* cn = cos_new.data<ov::float16>() + skip;
    const auto* sn = sin_new.data<ov::float16>() + skip;

    // Rows are independent; at the context limit this is ~200K of them.
    ov::parallel_for(num_tokens, [&](size_t t) {
        for (size_t j = 0; j < delta.half; ++j) {
            const float c_old = static_cast<float>(co[t * rotary_ndims + j]);
            const float s_old = static_cast<float>(so[t * rotary_ndims + j]);
            const float c_new = static_cast<float>(cn[t * rotary_ndims + j]);
            const float s_new = static_cast<float>(sn[t * rotary_ndims + j]);
            const float norm = c_old * c_old + s_old * s_old;
            delta.cos[t * delta.half + j] = (c_new * c_old + s_new * s_old) / norm;
            delta.sin[t * delta.half + j] = (s_new * c_old - c_new * s_old) / norm;
        }
    });
    return delta;
}

void ov::npuw::longrope::rerotate_keys(const ov::SoPtr<ov::ITensor>& tensor,
                                       uint32_t seq_dim,
                                       uint32_t num_tokens,
                                       const RegimeDelta& delta) {
    if (delta.half == 0u || num_tokens == 0u) {
        return;
    }

    const auto& shape = tensor->get_shape();
    const size_t rank = shape.size();
    OPENVINO_ASSERT(rank >= 2u && seq_dim + 1u < rank,
                    "NPUW: unexpected past-key layout for LongRoPE re-rotation: sequence axis ",
                    seq_dim,
                    " of a rank-",
                    rank,
                    " tensor.");
    const size_t head_dim = shape[rank - 1];
    OPENVINO_ASSERT(head_dim >= delta.half * 2u,
                    "NPUW: past-key head dimension ",
                    head_dim,
                    " is smaller than the rotary dimension ",
                    delta.half * 2u,
                    ".");
    const size_t seq_len = shape[seq_dim];
    OPENVINO_ASSERT(num_tokens <= seq_len, "NPUW: more cached tokens than the past-key tensor can hold.");

    // Rows are addressed arithmetically, which holds only for a densely packed tensor -
    // the layout every KV buffer NPUW hands out has.
    size_t seq_stride = 1u;
    for (size_t d = seq_dim + 1u; d < rank; ++d) {
        seq_stride *= shape[d];
    }
    OPENVINO_ASSERT(tensor->get_strides()[seq_dim] == seq_stride * tensor->get_element_type().size(),
                    "NPUW: past-key tensor is not densely packed, cannot re-rotate in place.");

    const size_t outer = tensor->get_size() / (seq_len * seq_stride);
    const size_t rows_per_token = seq_stride / head_dim;

    switch (tensor->get_element_type()) {
    case ov::element::f16: {
        // f16 goes through the SIMD kernel: expressed with ov::float16 the loop would
        // spend all its time in that class's out-of-line float conversions.
        auto* data = reinterpret_cast<uint16_t*>(tensor->data<ov::float16>());
        for (size_t o = 0; o < outer; ++o) {
            ov::npuw::util::XARCH::rerotate_f16_rows(data + o * seq_len * seq_stride,
                                                     num_tokens,
                                                     seq_stride,
                                                     rows_per_token,
                                                     head_dim,
                                                     delta.cos.data(),
                                                     delta.sin.data(),
                                                     delta.half);
        }
        break;
    }
    case ov::element::f32:
        rerotate_planes(tensor->data<float>(),
                        outer,
                        seq_len,
                        seq_stride,
                        rows_per_token,
                        head_dim,
                        num_tokens,
                        delta);
        break;
    default:
        // A quantized KV cache cannot be turned without dequantizing it first. Leaving
        // it in the previous regime is what happens today anyway, so warn instead of
        // failing an otherwise working configuration.
        LOG_WARN("LongRoPE key re-rotation does not support the KV cache element type "
                 << tensor->get_element_type() << "; cached keys stay in the previous regime.");
        break;
    }
}

void ov::npuw::longrope::rerotate_cached_keys(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                              const PortsMap& in_ports,
                                              const std::vector<std::string>& past_kv_names,
                                              ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                                              uint32_t seq_dim,
                                              uint32_t num_tokens,
                                              int64_t first_position_id,
                                              bool to_long) {
    const auto delta = make_regime_delta(tables, first_position_id, num_tokens, to_long);
    if (delta.half == 0u) {
        return;
    }

    LOG_DEBUG("Re-rotating " << num_tokens << " cached keys into the " << (to_long ? "long" : "short")
                             << "-factor LongRoPE regime.");

    ov::parallel_for(past_kv_names.size(), [&](size_t idx) {
        const auto& name = past_kv_names[idx];
        if (!ov::npuw::util::isPastKeyParam(name)) {
            return;
        }
        const auto port_it = in_ports.find(name);
        if (port_it == in_ports.end()) {
            return;
        }
        rerotate_keys(request->get_tensor(port_it->second), seq_dim, num_tokens, delta);
    });
}
