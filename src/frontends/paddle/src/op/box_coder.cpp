// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs box_coder(const NodeContext& node) {
    auto prior_box = node.get_input("PriorBox");
    auto target_box = node.get_input("TargetBox");
    const auto axis = node.get_attribute<int32_t>("axis", 0);
    const auto norm = node.get_attribute<bool>("box_normalized", true);
    const auto uns_axes = default_opset::Constant::create(ov::element::i64, {1}, {axis});

    Output<Node> variance;
    if (node.has_input("PriorBoxVar")) {
        variance = node.get_input("PriorBoxVar");
        variance = std::make_shared<default_opset::Unsqueeze>(variance, uns_axes);
    } else {
        const std::vector<float> var_vector = node.get_attribute<std::vector<float>>("variance", {1.0, 1.0, 1.0, 1.0});
        variance = default_opset::Constant::create(ov::element::f32, {1, 4}, var_vector);
    }
    const auto code_type = node.get_attribute<std::string>("code_type");
    PADDLE_OP_CHECK(node, (code_type == "decode_center_size"), "Currently only support decode mode!");

    const auto target_shape = std::make_shared<default_opset::ShapeOf>(target_box);
    prior_box = std::make_shared<default_opset::Unsqueeze>(prior_box, uns_axes);
    prior_box = std::make_shared<default_opset::Broadcast>(prior_box, target_shape);

    // split inputs into 4 elements
    auto split_axes = default_opset::Constant::create(element::i64, Shape{}, {2});
    const auto prior_split =
        std::make_shared<default_opset::Split>(prior_box, split_axes, 4);  // pxmin, pymin, pxmax, pymax
    const auto target_split = std::make_shared<default_opset::Split>(target_box, split_axes, 4);  // tx, ty, tw, th
    split_axes = default_opset::Constant::create(element::i64, Shape{}, {-1});
    const auto var_split = std::make_shared<default_opset::Split>(variance, split_axes, 4);  // pxv, pyv, pwv, phv
    OutputVector prior_out(4), target_out(4);

    const auto one = default_opset::Constant::create(ov::element::f32, {1}, {1});
    const auto two = default_opset::Constant::create(ov::element::f32, {1}, {2});

    // convert prior box from [xmin, ymin, xmax, ymax] to [x, y, w, h]
    prior_out[2] =
        std::make_shared<default_opset::Subtract>(prior_split->outputs()[2], prior_split->outputs()[0]);  // pw
    prior_out[3] =
        std::make_shared<default_opset::Subtract>(prior_split->outputs()[3], prior_split->outputs()[1]);  // ph
    if (!norm) {
        prior_out[2] = std::make_shared<default_opset::Add>(prior_out[2], one);
        prior_out[3] = std::make_shared<default_opset::Add>(prior_out[3], one);
    }
    prior_out[0] = std::make_shared<default_opset::Add>(prior_split->outputs()[0],
                                                        std::make_shared<default_opset::Divide>(prior_out[2], two));
    prior_out[1] = std::make_shared<default_opset::Add>(prior_split->outputs()[1],
                                                        std::make_shared<default_opset::Divide>(prior_out[3], two));

    for (int i = 0; i < 4; i++) {
        target_out[i] = target_split->outputs()[i];
    }

    OutputVector outputs(4), half_target(2), target_box_center(2);  // ox, oy, ow, oh  w / 2, h/ 2
    Output<Node> temp;
    const int offset = 2;
    for (int i = 0; i < 2; i++) {
        // get half_target
        temp = std::make_shared<default_opset::Multiply>(target_out[offset + i], var_split->outputs()[offset + i]);
        temp = std::make_shared<default_opset::Exp>(temp);
        temp = std::make_shared<default_opset::Multiply>(temp, prior_out[offset + i]);
        half_target[i] = std::make_shared<default_opset::Divide>(temp, two);
        // get target_box_center
        temp = std::make_shared<default_opset::Multiply>(prior_out[offset + i],
                                                         var_split->outputs()[i]);  // pw * pxv or ph * pyv
        temp = std::make_shared<default_opset::Multiply>(temp, target_out[i]);      // pw * pxv * tx or ph * pyv * ty
        target_box_center[i] =
            std::make_shared<default_opset::Add>(temp, prior_out[i]);  // px + pw * pxv * tx or py + ph * pyv * ty
    }

    outputs[0] = std::make_shared<default_opset::Subtract>(target_box_center[0], half_target[0]);
    outputs[1] = std::make_shared<default_opset::Subtract>(target_box_center[1], half_target[1]);
    outputs[2] = std::make_shared<default_opset::Add>(target_box_center[0], half_target[0]);
    outputs[3] = std::make_shared<default_opset::Add>(target_box_center[1], half_target[1]);
    if (!norm) {
        outputs[2] = std::make_shared<default_opset::Subtract>(outputs[2], one);
        outputs[3] = std::make_shared<default_opset::Subtract>(outputs[3], one);
    }

    return node.default_single_output_mapping({std::make_shared<default_opset::Concat>(outputs, -1)}, {"OutputBox"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
