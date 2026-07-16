// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_soft_max(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto input_node = context.get_input(0).get_node_shared_ptr();
    ov::Output<Node> res;

    float scale = context.get_attribute<float>("scale", 1.0f);
    float max_bias = context.get_attribute<float>("max_bias", 0.0f);
    const uint32_t n_head = context.get_input(0).get_partial_shape().get_shape()[0];

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    auto scaled_input = std::make_shared<ov::op::v1::Multiply>(input_node, scale_node);

    if (context.get_input_size() < 2) {
        res = std::make_shared<ov::op::v8::Softmax>(scaled_input, 2);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    ov::Output<ov::Node> mask_node_sliced;
    if (context.has_input("KQ_mask_sliced")) {
        mask_node_sliced = context.get_input("KQ_mask_sliced");
    } else {
        auto token_len = get_dimensions(input_node, {1});
        auto mask_node = context.get_input(1);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        mask_node_sliced = std::make_shared<ov::op::v8::Slice>(mask_node, zero, token_len, one, one);
    }

    auto output_type = context.get_attribute<ov::element::Type>("output_type");
    if (mask_node_sliced.get_element_type() != output_type) {
        mask_node_sliced = std::make_shared<ov::op::v0::Convert>(mask_node_sliced, output_type);
    }

    ov::Output<ov::Node> biased_input = scaled_input;
    if (max_bias > 0.0f) {
        // ALiBi: per-head slope[h] applied to the mask (ggml ggml_compute_forward_soft_max_f32).
        const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(n_head)));
        const float m0 = std::pow(2.0f, -max_bias / n_head_log2);
        const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
        std::vector<float> slopes(n_head);
        for (uint32_t h = 0; h < n_head; ++h) {
            slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<float>(h + 1)) : std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
        }
        auto slope_node = std::make_shared<ov::op::v0::Constant>(output_type,
                                                                 ov::Shape{n_head, 1, 1},
                                                                 slopes);
        auto slope_mask = std::make_shared<ov::op::v1::Multiply>(mask_node_sliced, slope_node);
        biased_input = std::make_shared<ov::op::v1::Add>(scaled_input, slope_mask);
    } else {
        biased_input = std::make_shared<ov::op::v1::Add>(scaled_input, mask_node_sliced);
    }

    res = std::make_shared<ov::op::v8::Softmax>(biased_input, 2);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
