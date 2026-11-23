// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/common.hpp"

#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

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

std::string layer_norm(GraphEmitter& e,
                       const std::string& in,
                       const std::string& weight,
                       const std::string& out_prefix,
                       float eps) {
    e.add_weight(weight);
    auto norm = e.add_op("GGML_OP_NORM", out_prefix + ".norm", {in}, 0, {{"eps", eps}});
    auto scaled = e.add_op("GGML_OP_MUL", out_prefix + ".scaled", {norm, weight});
    return e.has_weight(bias_weight_name(weight)) ? add_bias(e, scaled, bias_weight_name(weight), out_prefix) : scaled;
}

std::string bias_weight_name(const std::string& weight_name) {
    return strip_weight_suffix(weight_name) + ".bias";
}

std::string scale(GraphEmitter& e, const std::string& x, float factor, const std::string& name) {
    return e.add_op("GGML_OP_SCALE", name, {x}, 0, {{"scale", factor}, {"bias", 0.0f}});
}

std::string reshape(GraphEmitter& e, const std::string& x, std::vector<int64_t> target, const std::string& name) {
    return e.add_op("GGML_OP_RESHAPE", name, {x}, 6, {{"reshape_target", std::move(target)}, {"special_zero", true}});
}

std::string add_bias(GraphEmitter& e, const std::string& x, const std::string& bias_weight, const std::string& name) {
    e.add_weight(bias_weight);
    return e.add_op("GGML_OP_ADD", name, {x, bias_weight});
}

std::array<std::string, 2> split_interleaved_projection(GraphEmitter& e,
                                                        const std::string& input,
                                                        const std::array<std::string, 2>& names,
                                                        int64_t groups,
                                                        int64_t width) {
    const auto rank = e.value(input).get_partial_shape().rank().get_length();
    std::vector<int64_t> grouped_shape(rank - 1, 0);
    grouped_shape.insert(grouped_shape.end(), {groups, 2 * width});
    const auto grouped = e.add_op("GGML_OP_RESHAPE",
                                  input + "_grouped",
                                  {input},
                                  6,
                                  {{"reshape_target", grouped_shape}, {"special_zero", true}});
    std::vector<int64_t> output_shape(rank - 1, 0);
    output_shape.push_back(groups * width);
    std::array<std::string, 2> outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto slice =
            e.add_op("GGML_OP_VIEW",
                     names[i] + "_slice",
                     {grouped},
                     3,
                     {{"view_slice", std::vector<int64_t>{rank, static_cast<int64_t>(i) * width, width}}});
        outputs[i] = e.add_op("GGML_OP_RESHAPE",
                              names[i],
                              {slice},
                              6,
                              {{"reshape_target", output_shape}, {"special_zero", true}});
    }
    return outputs;
}

std::string concat_weight_rows(GraphEmitter& e, const std::string& name, const std::vector<std::string>& weights) {
    OPENVINO_ASSERT(!weights.empty(), "[GGUF] weight concatenation requires inputs");
    const auto rank = e.weight_shape(weights.front()).size();
    OPENVINO_ASSERT(rank > 0 && weights.size() > 1, "[GGUF] weight concatenation requires nonscalar inputs");
    auto input = weights.front();
    for (size_t i = 1; i < weights.size(); ++i) {
        const auto output = i + 1 == weights.size() ? name : name + ".concat" + std::to_string(i);
        e.derive_weight(output,
                        "GGML_OP_CONCAT",
                        {input, weights[i]},
                        0,
                        {{"concat_axis", static_cast<int>(rank - 1)}});
        input = output;
    }
    return name;
}

}  // namespace ov::frontend::gguf::blocks
