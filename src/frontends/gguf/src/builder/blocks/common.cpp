// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/common.hpp"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

namespace ov::frontend::gguf::blocks {

std::string rms_norm(GraphEmitter& e,
                     const std::string& in,
                     const std::string& weight,
                     const std::string& out_prefix,
                     float eps) {
    e.add_weight(weight);
    auto norm = e.add_op("GGML_OP_RMS_NORM", out_prefix + ".rms", {in}, 0, {{"eps", eps}});
    return e.add_op("GGML_OP_MUL", out_prefix, {norm, weight});
}

std::string scale(GraphEmitter& e, const std::string& x, float factor, const std::string& name) {
    return e.add_op("GGML_OP_SCALE", name, {x}, 0, {{"scale", factor}, {"bias", 0.0f}});
}

std::string add_bias(GraphEmitter& e, const std::string& x, const std::string& bias_weight, const std::string& name) {
    e.add_named_weight(bias_weight);
    return e.add_op("GGML_OP_ADD", name, {x, bias_weight});
}

WeightTensors weight_parts(GraphEmitter& e, const std::string& base) {
    auto& w = e.weights();
    const auto get = [&](const char* suffix) {
        const auto it = w.find(base + suffix);
        return it == w.end() ? ov::Tensor() : it->second;
    };
    return {get(".weight"), get(".scales"), get(".zp")};
}

GgufTensorType weight_qtype(GraphEmitter& e, const std::string& base) {
    const auto it = e.qtypes().find(base + ".qtype");
    return it == e.qtypes().end() ? GGUF_TYPE_F16 : it->second;
}

void store_parts(GraphEmitter& e, const std::string& base, const WeightTensors& t, GgufTensorType qtype) {
    auto& w = e.weights();
    w[base + ".weight"] = t.weight;
    if (t.scales) {
        w[base + ".scales"] = t.scales;
    }
    if (t.zero_point) {
        w[base + ".zp"] = t.zero_point;
    }
    e.qtypes()[base + ".qtype"] = qtype;
}

// Concatenate the rows of two weights with the same quantization layout into one; empty on mismatch.
std::optional<WeightTensors> concat_rows(const WeightTensors& a,
                                         const WeightTensors& b,
                                         GgufTensorType qa,
                                         GgufTensorType qb) {
    if (qa != qb || !a.weight || !b.weight) {
        return std::nullopt;
    }
    WeightTensors out;
    const std::vector<std::tuple<const ov::Tensor*, const ov::Tensor*, ov::Tensor*>> parts{
        {&a.weight, &b.weight, &out.weight},
        {&a.scales, &b.scales, &out.scales},
        {&a.zero_point, &b.zero_point, &out.zero_point}};
    for (const auto& [x, y, dst] : parts) {
        if (static_cast<bool>(*x) != static_cast<bool>(*y)) {
            return std::nullopt;
        }
        if (!*x) {
            continue;
        }
        auto sx = x->get_shape(), sy = y->get_shape();
        if (x->get_element_type() != y->get_element_type() || sx.empty() || sx.size() != sy.size() ||
            !std::equal(sx.begin() + 1, sx.end(), sy.begin() + 1)) {
            return std::nullopt;
        }
        sx[0] += sy[0];
        *dst = ov::Tensor(x->get_element_type(), sx);
        std::memcpy(dst->data(), x->data(), x->get_byte_size());
        std::memcpy(static_cast<uint8_t*>(dst->data()) + x->get_byte_size(), y->data(), y->get_byte_size());
    }
    return out;
}

namespace {

bool is_symmetric_32(GgufTensorType q) {
    return q == GGUF_TYPE_Q4_0 || q == GGUF_TYPE_Q8_0;
}

bool is_asymmetric_32(GgufTensorType q) {
    return q == GGUF_TYPE_Q4_1 || q == GGUF_TYPE_Q4_K || q == GGUF_TYPE_Q5_K;
}

// Exactly re-express a 32-group weight in the 8-bit layout of `target` (Q8_0 or Q5_K).
std::optional<WeightTensors> widen_to_8bit(const WeightTensors& t, GgufTensorType qtype, GgufTensorType target) {
    const auto& ws = t.weight.get_shape();
    const bool source_ok = is_symmetric_32(qtype) || (is_asymmetric_32(qtype) && target == GGUF_TYPE_Q5_K);
    if (!source_ok || !t.weight || !t.scales || ws.size() != 2) {
        return std::nullopt;
    }
    const size_t rows = ws[0];
    const size_t cols = t.scales.get_shape().back() * 32;
    const auto src_et = t.weight.get_element_type();
    const size_t bits = (src_et == ov::element::i4 || src_et == ov::element::u32) ? 4 : 8;
    if (t.weight.get_byte_size() * 8 != rows * cols * bits) {
        return std::nullopt;
    }
    const auto* src = static_cast<const uint8_t*>(t.weight.data());
    const bool to_unsigned = target == GGUF_TYPE_Q5_K;

    WeightTensors out;
    out.scales = t.scales;
    out.weight = ov::Tensor(to_unsigned ? ov::element::u8 : ov::element::i8, ov::Shape{rows, cols});
    auto* dst = static_cast<uint8_t*>(out.weight.data());
    // Signed sources are shifted by 128 into u8 with a zero-point of 128.
    const int shift = to_unsigned && is_symmetric_32(qtype) ? 128 : 0;
    const size_t n = rows * cols;
    if (src_et == ov::element::i4 || src_et == ov::element::u32) {
        const bool is_signed = src_et == ov::element::i4;
        for (size_t k = 0; k < n; ++k) {
            int v = (src[k / 2] >> (4 * (k % 2))) & 0xF;
            if (is_signed && v >= 8) {
                v -= 16;
            }
            dst[k] = static_cast<uint8_t>(v + shift);
        }
    } else if (src_et == ov::element::i8 || src_et == ov::element::u8) {
        for (size_t k = 0; k < n; ++k) {
            const int v = src_et == ov::element::i8 ? static_cast<int8_t>(src[k]) : src[k];
            dst[k] = static_cast<uint8_t>(v + shift);
        }
    } else {
        return std::nullopt;
    }

    if (!to_unsigned) {
        return out;
    }
    if (t.zero_point) {
        if (t.zero_point.get_element_type() != ov::element::u8) {
            return std::nullopt;
        }
        out.zero_point = t.zero_point;
    } else {
        out.zero_point = ov::Tensor(ov::element::u8, t.scales.get_shape());
        std::memset(out.zero_point.data(), 128, out.zero_point.get_byte_size());
    }
    return out;
}

}  // namespace

std::optional<std::pair<WeightTensors, GgufTensorType>> concat_rows_widened(const WeightTensors& a,
                                                                            const WeightTensors& b,
                                                                            GgufTensorType qa,
                                                                            GgufTensorType qb) {
    if (auto same = concat_rows(a, b, qa, qb)) {
        return std::make_pair(std::move(*same), qa);
    }
    const auto supported = [](GgufTensorType q) {
        return is_symmetric_32(q) || is_asymmetric_32(q);
    };
    if (!supported(qa) || !supported(qb)) {
        return std::nullopt;
    }
    const auto target = is_symmetric_32(qa) && is_symmetric_32(qb) ? GGUF_TYPE_Q8_0 : GGUF_TYPE_Q5_K;
    auto wa = widen_to_8bit(a, qa, target);
    auto wb = widen_to_8bit(b, qb, target);
    if (!wa || !wb) {
        return std::nullopt;
    }
    auto merged = concat_rows(*wa, *wb, target, target);
    if (!merged) {
        return std::nullopt;
    }
    return std::make_pair(std::move(*merged), target);
}

}  // namespace ov::frontend::gguf::blocks
