// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate_diff.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pad(NodeContext& context) {
    auto data = context.get_input(0);
    auto paddings = context.const_input<std::vector<int64_t>>(1);
    std::string mode = "constant";
    auto shape = context.mark_node(std::make_shared<opset8::ShapeOf>(data, element::i32));
    auto rank = context.mark_node(std::make_shared<opset8::ShapeOf>(shape, element::i32));
    auto reduced_rank = context.mark_node(std::make_shared<opset8::Squeeze>(rank));
    auto zero = context.mark_node(opset8::Constant::create(element::i32, Shape{}, {0}));
    auto zero_f = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto pad_size_half = paddings.size() / 2;
    std::vector<int64_t> pad_b(pad_size_half, 0);
    std::vector<int64_t> pad_e(pad_size_half, 0);
    auto pad_mode = ov::op::PadMode::CONSTANT;
    for (int i = 0; i < pad_size_half; i++) {
        pad_b[i] = paddings[paddings.size() - 2 - 2 * i];
        pad_e[i] = paddings[paddings.size() - 1 - 2 * i];
    }
    auto pads_begin_short = context.mark_node(opset8::Constant::create(element::i32, Shape{pad_size_half}, pad_b));
    auto pads_end_short = context.mark_node(opset8::Constant::create(element::i32, Shape{pad_size_half}, pad_e));
    auto pads_short_len = context.mark_node(opset8::Constant::create(element::i32, Shape{1}, {pad_size_half}));
    auto pads_diff = context.mark_node(std::make_shared<opset8::Subtract>(rank, pads_short_len));
    auto pads_remaining = context.mark_node(std::make_shared<opset8::Broadcast>(zero, pads_diff));
    auto pads_begins =
        context.mark_node(std::make_shared<opset8::Concat>(NodeVector{pads_remaining, pads_begin_short}, 0));
    auto pads_ends = context.mark_node(std::make_shared<opset8::Concat>(NodeVector{pads_remaining, pads_end_short}, 0));
    if (!context.input_is_none(2)) {
        mode = context.const_input<std::string>(2);
    }
    if (mode == "circular") {
        int64_t pad_l;
        int64_t pad_r;
        auto pad_last_id = paddings.size();
        auto cur = data.get_node_shared_ptr();
        auto step = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));
        for (auto i = 0; i < pad_size_half; i++) {
            ov::NodeVector tensors;
            pad_r = paddings[pad_last_id - (2 * i + 1)];
            pad_l = paddings[pad_last_id - (2 * i + 2)];
            auto axes = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {2 + i}));
            if (pad_l > 0) {
                auto start =
                    context.mark_node(context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-pad_l})));
                auto end = context.mark_node(std::make_shared<opset8::Gather>(
                    shape,
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {2 + i})),
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}))));

                auto left = context.mark_node(std::make_shared<opset8::Slice>(cur, start, end, step, axes));
                tensors.push_back(left);
            }
            if (pad_l < 0 || pad_r < 0) {
                auto start = context.mark_node(
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {pad_l < 0 ? -pad_l : 0})));
                auto end = context.mark_node(
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {pad_r < 0 ? pad_r : 0})));
                auto middle = context.mark_node(std::make_shared<opset8::Slice>(cur, start, end, step, axes));
                tensors.push_back(middle);
            } else {
                tensors.push_back(cur);
            }
            if (pad_r > 0) {
                auto start = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
                auto end = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {pad_r}));
                auto right = context.mark_node(std::make_shared<opset8::Slice>(cur, start, end, step, axes));
                tensors.push_back(right);
            }
            if (tensors.size()) {
                cur = context.mark_node(std::make_shared<opset8::Concat>(tensors, 2 + i));
            }
        }
        return {cur};
    }
    if (mode == "constant") {
        if (!context.input_is_none(3)) {
            auto pad_value = context.get_input(3);
            return {context.mark_node(
                std::make_shared<opset8::Pad>(data, pads_begins, pads_ends, pad_value, ov::op::PadMode::CONSTANT))};
        }
        return {context.mark_node(
            std::make_shared<opset8::Pad>(data, pads_begins, pads_ends, zero_f, ov::op::PadMode::CONSTANT))};
    }
    if (mode == "reflect") {
        return {context.mark_node(
            std::make_shared<opset8::Pad>(data, pads_begins, pads_ends, zero_f, ov::op::PadMode::REFLECT))};
    }
    if (mode == "replicate") {
        return {context.mark_node(
            std::make_shared<opset8::Pad>(data, pads_begins, pads_ends, zero_f, ov::op::PadMode::EDGE))};
    }

    FRONT_END_OP_CONVERSION_CHECK(false, "aten::pad conversion doesn't support [ " + mode + " ] padding mode");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov