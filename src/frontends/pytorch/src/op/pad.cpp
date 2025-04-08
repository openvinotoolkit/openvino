// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector translate_pad_common(const NodeContext& context,
                                  const Output<Node>& data,
                                  const std::vector<int64_t>& paddings,
                                  const Output<Node>& pad_value,
                                  const std::string& mode = "constant") {
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, data);
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    size_t pad_size_half = paddings.size() / 2;
    std::vector<int64_t> pad_b(pad_size_half, 0);
    std::vector<int64_t> pad_e(pad_size_half, 0);
    for (size_t i = 0; i < pad_size_half; i++) {
        pad_b[i] = paddings[paddings.size() - 2 - 2 * i];
        pad_e[i] = paddings[paddings.size() - 1 - 2 * i];
    }
    auto pads_begin_short = context.mark_node(v0::Constant::create(element::i32, Shape{pad_size_half}, pad_b));
    auto pads_end_short = context.mark_node(v0::Constant::create(element::i32, Shape{pad_size_half}, pad_e));
    auto pads_short_len = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {pad_size_half}));
    auto pads_diff = context.mark_node(std::make_shared<v1::Subtract>(rank, pads_short_len));
    auto pads_remaining = context.mark_node(std::make_shared<v3::Broadcast>(zero, pads_diff));
    auto pads_begins = context.mark_node(std::make_shared<v0::Concat>(NodeVector{pads_remaining, pads_begin_short}, 0));
    auto pads_ends = context.mark_node(std::make_shared<v0::Concat>(NodeVector{pads_remaining, pads_end_short}, 0));
    if (mode == "circular") {
        int64_t pad_l;
        int64_t pad_r;
        auto pad_last_id = paddings.size();
        auto cur = data;
        auto step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
        auto zero_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        for (size_t i = 0; i < pad_size_half; i++) {
            OutputVector tensors;
            pad_r = paddings[pad_last_id - (2 * i + 1)];
            pad_l = paddings[pad_last_id - (2 * i + 2)];
            auto axes = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2 + i}));
            if (pad_l > 0) {
                auto start = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-pad_l}));
                auto end = context.mark_node(std::make_shared<v8::Gather>(shape, axes, zero_1d));

                auto left = context.mark_node(std::make_shared<v8::Slice>(cur, start, end, step, axes));
                tensors.push_back(left);
            }
            if (pad_l < 0 || pad_r < 0) {
                auto start = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {pad_l < 0 ? -pad_l : 0}));
                auto end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {pad_r < 0 ? pad_r : 0}));
                auto middle = context.mark_node(std::make_shared<v8::Slice>(cur, start, end, step, axes));
                tensors.push_back(middle);
            } else {
                tensors.push_back(cur);
            }
            if (pad_r > 0) {
                auto end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {pad_r}));
                auto right = context.mark_node(std::make_shared<v8::Slice>(cur, zero_1d, end, step, axes));
                tensors.push_back(right);
            }
            if (tensors.size()) {
                cur = context.mark_node(std::make_shared<v0::Concat>(tensors, 2 + i));
            }
        }
        return {cur};
    }
    auto pad_value_ = context.mark_node(std::make_shared<v1::ConvertLike>(pad_value, data));
    static const std::map<std::string, PadMode> pt_to_ov_pad{
        {"constant", PadMode::CONSTANT},
        {"reflect", PadMode::REFLECT},
        {"replicate", PadMode::EDGE},
    };
    auto ov_mode = pt_to_ov_pad.find(mode);
    PYTORCH_OP_CONVERSION_CHECK(ov_mode != pt_to_ov_pad.end(),
                                "aten::pad conversion doesn't support [ ",
                                mode,
                                " ] padding mode");
    return {context.mark_node(std::make_shared<v1::Pad>(data, pads_begins, pads_ends, pad_value_, ov_mode->second))};
}
}  // namespace

OutputVector translate_pad(const NodeContext& context) {
    num_inputs_check(context, 2, 4);
    auto data = context.get_input(0);
    auto paddings = context.const_input<std::vector<int64_t>>(1);
    std::string mode = "constant";

    if (!context.input_is_none(2)) {
        mode = context.const_input<std::string>(2);
    }

    Output<Node> pad_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    if (mode == "constant") {
        if (!context.input_is_none(3)) {
            pad_value = context.get_input(3);
        }
    }
    return translate_pad_common(context, data, paddings, pad_value, mode);
}

OutputVector translate_constant_pad_nd_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    auto paddings = context.const_input<std::vector<int64_t>>(1);
    auto pad_value = context.get_input(2);
    return translate_pad_common(context, data, paddings, pad_value);
}

OutputVector translate_reflection_pad_nd_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto data = context.get_input(0);
    auto paddings = context.const_input<std::vector<int64_t>>(1);
    Output<Node> pad_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    return translate_pad_common(context, data, paddings, pad_value, "reflect");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov