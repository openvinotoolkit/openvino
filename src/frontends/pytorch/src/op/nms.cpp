// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset9.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_nms(NodeContext& context) {
    auto const_0 = context.mark_node(opset9::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(opset9::Constant::create(element::i64, Shape{}, {1}));
    auto const_2 = context.mark_node(opset9::Constant::create(element::i64, Shape{1}, {2}));
    // the shape that is required by PyTorch operator differs from the shape required in OpenVino
    auto boxes_shape = context.mark_node(opset9::Constant::create(element::i64, Shape{3}, {1, -1, 4}));

    auto boxes = context.mark_node(std::make_shared<opset9::Reshape>(context.get_input(0), boxes_shape, false));
    // Unsqueeze operator is also used to align shapes required by PyTorch and OpenVino
    auto axis_01 = context.mark_node(opset9::Constant::create(element::i64, Shape{2}, {0, 1}));
    auto scores = context.mark_node(std::make_shared<opset9::Unsqueeze>(context.get_input(1), axis_01));
    auto max_output_per_class =
        context.mark_node(opset9::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()}));
    auto iou_threshold = context.get_input(2);

    auto nms_out = context.mark_node(
        std::make_shared<opset9::NonMaxSuppression>(boxes, scores, max_output_per_class, iou_threshold));
    auto select = context.mark_node(std::make_shared<opset9::Gather>(nms_out, const_2, const_1));
    auto squeeze = std::make_shared<opset9::Squeeze>(select, const_1);

    return {context.mark_node(squeeze)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
