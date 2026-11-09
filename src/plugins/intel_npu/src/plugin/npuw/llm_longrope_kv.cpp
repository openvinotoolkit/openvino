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
                     const float* delta_cos,
                     const float* delta_sin,
                     size_t half) {
    for (size_t o = 0; o < outer; ++o) {
        T* plane = data + o * seq_len * seq_stride;
        for (size_t t = 0; t < num_tokens; ++t) {
            const float* dcos = delta_cos + t * half;
            const float* dsin = delta_sin + t * half;
            T* token = plane + t * seq_stride;
            for (size_t r = 0; r < rows_per_token; ++r) {
                rerotate_row(token + r * head_dim, dcos, dsin, half);
            }
        }
    }
}

}  // anonymous namespace

ov::npuw::longrope::ModeDelta ov::npuw::longrope::make_mode_delta(
    ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
    int64_t first_position_id,
    uint32_t num_tokens,
    bool to_long) {
    ModeDelta delta;
    if (num_tokens == 0u || !tables.has_long) {
        // Without a distinct long-factor set both modes share the same coefficients,
        // so a flip changes nothing about the cached keys.
        return delta;
    }
    OPENVINO_ASSERT(tables.is_valid(), "NPUW: LongRoPE mode flipped but the cos/sin tables are missing.");

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

    // Row p of either mode holds the coefficients of absolute position p.
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

void ov::npuw::longrope::rerotate_keys(const KeyTensorLayout& layout,
                                       uint32_t num_tokens,
                                       const ModeDelta& delta,
                                       size_t delta_row_offset) {
    if (delta.half == 0u || num_tokens == 0u) {
        return;
    }
    OPENVINO_ASSERT((delta_row_offset + num_tokens) * delta.half <= delta.cos.size(),
                    "NPUW: LongRoPE delta rows [",
                    delta_row_offset,
                    ", ",
                    delta_row_offset + num_tokens,
                    ") fall outside the delta built for this transition.");
    const float* delta_cos = delta.cos.data() + delta_row_offset * delta.half;
    const float* delta_sin = delta.sin.data() + delta_row_offset * delta.half;
    if (layout.type == ov::element::f16) {
        // f16 goes through the SIMD kernel: expressed with ov::float16 the loop would
        // spend all its time in that class's out-of-line float conversions.
        auto* data = static_cast<uint16_t*>(layout.data);
        for (size_t o = 0; o < layout.outer; ++o) {
            ov::npuw::util::XARCH::rerotate_f16_rows(data + o * layout.seq_len * layout.seq_stride,
                                                     num_tokens,
                                                     layout.seq_stride,
                                                     layout.rows_per_token,
                                                     layout.head_dim,
                                                     delta_cos,
                                                     delta_sin,
                                                     delta.half);
        }
    } else {
        rerotate_planes(static_cast<float*>(layout.data),
                        layout.outer,
                        layout.seq_len,
                        layout.seq_stride,
                        layout.rows_per_token,
                        layout.head_dim,
                        num_tokens,
                        delta_cos,
                        delta_sin,
                        delta.half);
    }
}

ov::npuw::longrope::KeyTensorLayout ov::npuw::longrope::check_key_tensor(const ov::SoPtr<ov::ITensor>& tensor,
                                                                         uint32_t seq_dim,
                                                                         uint32_t num_tokens,
                                                                         const ModeDelta& delta) {
    OPENVINO_ASSERT(tensor, "NPUW: a past-key input of a LongRoPE model has no tensor bound to it.");

    const auto type = tensor->get_element_type();
    // A quantized cache cannot be turned without dequantizing it first, and no other
    // element type reaches the kernels below.
    OPENVINO_ASSERT(type == ov::element::f16 || type == ov::element::f32,
                    "NPUW: a LongRoPE mode change cannot re-rotate a KV cache of element type ",
                    type,
                    "; only f16 and f32 caches can be turned in place.");

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

    // Rows are addressed arithmetically from a single base pointer, which holds only
    // for a canonically packed tensor. Checking the sequence stride alone is not
    // enough: a view that crops the sequence axis keeps its parent's outer strides, so
    // every plane past the first would be read at the wrong offset. Demand the whole
    // row-major stride vector instead.
    const auto& strides = tensor->get_strides();
    OPENVINO_ASSERT(strides.size() == rank,
                    "NPUW: past-key tensor reports ",
                    strides.size(),
                    " strides for rank ",
                    rank,
                    ".");
    size_t dense_stride = type.size();
    for (size_t d = rank; d-- > 0;) {
        OPENVINO_ASSERT(strides[d] == dense_stride,
                        "NPUW: past-key tensor is not densely packed (stride ",
                        strides[d],
                        " at axis ",
                        d,
                        ", expected ",
                        dense_stride,
                        "), cannot re-rotate in place.");
        dense_stride *= shape[d];
    }

    size_t seq_stride = 1u;
    for (size_t d = seq_dim + 1u; d < rank; ++d) {
        seq_stride *= shape[d];
    }

    KeyTensorLayout layout;
    // Resolving the pointer here is itself a check: a device-side tensor that cannot be
    // mapped to the host throws, and it does so before any other layer was touched.
    layout.data = tensor->data();
    layout.type = type;
    layout.outer = tensor->get_size() / (seq_len * seq_stride);
    layout.seq_len = seq_len;
    layout.seq_stride = seq_stride;
    layout.rows_per_token = seq_stride / head_dim;
    layout.head_dim = head_dim;
    return layout;
}

void ov::npuw::longrope::rerotate_keys(const ov::SoPtr<ov::ITensor>& tensor,
                                       uint32_t seq_dim,
                                       uint32_t num_tokens,
                                       const ModeDelta& delta) {
    if (delta.half == 0u || num_tokens == 0u) {
        return;
    }
    rerotate_keys(check_key_tensor(tensor, seq_dim, num_tokens, delta), num_tokens, delta);
}

void ov::npuw::longrope::rerotate_cached_keys(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                              const PortsMap& in_ports,
                                              const std::vector<std::string>& past_kv_names,
                                              ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                                              uint32_t seq_dim,
                                              uint32_t num_tokens,
                                              int64_t first_position_id,
                                              bool to_long) {
    const auto delta = make_mode_delta(tables, first_position_id, num_tokens, to_long);
    if (delta.half == 0u) {
        return;
    }

    // Resolve and check every live key first. Anything that throws does so here, with
    // the cache still wholly in the previous mode and the caller's mode flag not
    // yet advanced.
    std::vector<KeyTensorLayout> layouts;
    layouts.reserve(past_kv_names.size());
    for (const auto& name : past_kv_names) {
        if (!ov::npuw::util::isPastKeyParam(name)) {
            continue;
        }
        const auto port_it = in_ports.find(name);
        OPENVINO_ASSERT(port_it != in_ports.end(),
                        "NPUW: past-key input ",
                        name,
                        " is missing from the request the LongRoPE mode change has to re-rotate.");
        layouts.push_back(check_key_tensor(request->get_tensor(port_it->second), seq_dim, num_tokens, delta));
    }

    LOG_DEBUG("Re-rotating " << num_tokens << " cached keys of " << layouts.size() << " layers into the "
                             << (to_long ? "long" : "short") << "-factor LongRoPE mode.");

    ov::parallel_for(layouts.size(), [&](size_t idx) {
        rerotate_keys(layouts[idx], num_tokens, delta);
    });
}

void ov::npuw::longrope::rerotate_cached_key_blocks(const std::vector<KeyBlock>& blocks,
                                                    ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                                                    uint32_t seq_dim,
                                                    uint32_t num_cached_tokens,
                                                    int64_t first_position_id,
                                                    bool to_long) {
    // One delta for the whole conversation; each block reads the rows its own positions
    // land on, so the blocks need not be contiguous or in any particular order.
    const auto delta = make_mode_delta(tables, first_position_id, num_cached_tokens, to_long);
    if (delta.half == 0u) {
        return;
    }

    std::vector<KeyTensorLayout> layouts;
    layouts.reserve(blocks.size());
    for (const auto& block : blocks) {
        OPENVINO_ASSERT(static_cast<size_t>(block.first_token) + block.num_tokens <= num_cached_tokens,
                        "NPUW: a key block covering tokens [",
                        block.first_token,
                        ", ",
                        block.first_token + block.num_tokens,
                        ") reaches past the ",
                        num_cached_tokens,
                        " tokens the LongRoPE mode change was given.");
        layouts.push_back(check_key_tensor(block.tensor, seq_dim, block.num_tokens, delta));
    }

    LOG_DEBUG("Re-rotating " << num_cached_tokens << " cached keys held in " << layouts.size() << " blocks into the "
                             << (to_long ? "long" : "short") << "-factor LongRoPE mode.");

    ov::parallel_for(layouts.size(), [&](size_t idx) {
        rerotate_keys(layouts[idx], blocks[idx].num_tokens, delta, blocks[idx].first_token);
    });
}
