// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/common.hpp"

#include <cstring>
#include <tuple>
#include <vector>

namespace ov::frontend::gguf::blocks {

std::string rms_norm(GraphEmitter& e,
                     const std::string& in,
                     const std::string& weight,
                     const std::string& out_prefix,
                     float eps) {
    if (strip_weight_suffix(weight) == weight)
        e.add_named_weight(weight);
    else
        e.add_weight(weight);
    auto norm = e.add_op("GGML_OP_RMS_NORM", out_prefix + ".rms", {in}, 0, {{"eps", eps}});
    return e.add_op("GGML_OP_MUL", out_prefix, {norm, weight});
}

std::string layer_norm(GraphEmitter& e,
                       const std::string& in,
                       const std::string& weight,
                       const std::string& out_prefix,
                       float eps) {
    e.add_weight(weight);
    auto norm = e.add_op("GGML_OP_NORM", out_prefix + ".norm", {in}, 0, {{"eps", eps}});
    auto scaled = e.add_op("GGML_OP_MUL", out_prefix + ".scaled", {norm, weight});
    return add_bias(e, scaled, bias_weight_name(weight), out_prefix);
}

std::string bias_weight_name(const std::string& weight_name) {
    return strip_weight_suffix(weight_name) + ".bias";
}

std::string scale(GraphEmitter& e, const std::string& x, float factor, const std::string& name) {
    return e.add_op("GGML_OP_SCALE", name, {x}, 0, {{"scale", factor}, {"bias", 0.0f}});
}

std::string add_bias(GraphEmitter& e, const std::string& x, const std::string& bias_weight, const std::string& name) {
    e.add_named_weight(bias_weight);
    return e.add_op("GGML_OP_ADD", name, {x, bias_weight});
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

}  // namespace ov::frontend::gguf::blocks
