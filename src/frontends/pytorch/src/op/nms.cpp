// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_nms(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
    // the shape that is required by PyTorch operator differs from the shape required in OpenVino
    auto boxes_shape = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {1, -1, 4}));

    auto boxes = context.mark_node(std::make_shared<v1::Reshape>(context.get_input(0), boxes_shape, false));
    // Unsqueeze operator is also used to align shapes required by PyTorch and OpenVino
    auto axis_01 = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {0, 1}));
    auto scores = context.mark_node(std::make_shared<v0::Unsqueeze>(context.get_input(1), axis_01));
    auto max_output_per_class =
        context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()}));
    auto iou_threshold = context.get_input(2);

    auto score_threshold =
        context.mark_node(v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()}));
    auto nms_out = context.mark_node(
        std::make_shared<v9::NonMaxSuppression>(boxes, scores, max_output_per_class, iou_threshold, score_threshold));
    auto select = context.mark_node(std::make_shared<v8::Gather>(nms_out, const_2, const_1));

    return {context.mark_node(std::make_shared<v0::Squeeze>(select, const_1))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
