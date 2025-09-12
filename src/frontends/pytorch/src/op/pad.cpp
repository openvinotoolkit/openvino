// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
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
                                  const Output<Node>& paddings,
                                  const Output<Node>& pad_value,
                                  const std::string& mode = "constant") {
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, data);
    if (mode == "circular") {
        int64_t pad_l;
        int64_t pad_r;
        auto paddings_const = ov::util::get_constant_from_source(paddings);
        PYTORCH_OP_CONVERSION_CHECK(paddings_const,
                                    "aten::pad conversion for circular mode supports only constant paddings");
        auto paddings_data = paddings_const->cast_vector<int64_t>();
        auto pad_last_id = paddings_data.size();
        auto pad_size_half = pad_last_id / 2;
        auto cur = data;
        auto step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
        auto zero_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        for (size_t i = 0; i < pad_size_half; i++) {
            OutputVector tensors;
            pad_r = paddings_data[pad_last_id - (2 * i + 1)];
            pad_l = paddings_data[pad_last_id - (2 * i + 2)];
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
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));

    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto pads_shape = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {-1, 2}));
    auto pads_reshape = context.mark_node(std::make_shared<v1::Reshape>(paddings, pads_shape, false));
    auto pads_split = context.mark_node(std::make_shared<v1::Split>(pads_reshape, one, 2));
    auto pads_begin_short = context.mark_node(std::make_shared<v0::Squeeze>(pads_split->output(0), one));
    auto pads_end_short = context.mark_node(std::make_shared<v0::Squeeze>(pads_split->output(1), one));

    auto pads_short_len = context.mark_node(std::make_shared<v3::ShapeOf>(pads_begin_short, element::i32));
    pads_short_len = context.mark_node(std::make_shared<v0::Squeeze>(pads_short_len, zero));
    if (const auto c_node = ov::util::get_constant_from_source(pads_short_len)) {
        pads_short_len = c_node;
    }
    auto pads_start_idx = context.mark_node(std::make_shared<v1::Add>(pads_short_len, neg_one));
    auto pad_idx_range = context.mark_node(std::make_shared<v4::Range>(pads_start_idx, neg_one, neg_one, element::i32));
    pads_begin_short = context.mark_node(std::make_shared<v8::Gather>(pads_begin_short, pad_idx_range, zero));
    pads_end_short = context.mark_node(std::make_shared<v8::Gather>(pads_end_short, pad_idx_range, zero));

    if (const auto begins = ov::util::get_constant_from_source(pads_begin_short)) {
        pads_begin_short = begins;
    }
    if (const auto ends = ov::util::get_constant_from_source(pads_end_short)) {
        pads_end_short = ends;
    }

    auto input_rank = std::get<1>(get_shape_rank(context, data, false, element::i32));
    auto pads_diff = context.mark_node(std::make_shared<v1::Subtract>(input_rank, pads_short_len));
    auto pads_remaining = context.mark_node(std::make_shared<v3::Broadcast>(zero, pads_diff));
    auto pads_remaining_c = context.mark_node(std::make_shared<v1::ConvertLike>(pads_remaining, paddings));

    auto pads_begin =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{pads_remaining_c, pads_begin_short}, 0));
    auto pads_end = context.mark_node(std::make_shared<v0::Concat>(OutputVector{pads_remaining_c, pads_end_short}, 0));

    if (const auto begins = ov::util::get_constant_from_source(pads_begin)) {
        pads_begin = begins;
    }
    if (const auto ends = ov::util::get_constant_from_source(pads_end)) {
        pads_end = ends;
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
    return {context.mark_node(std::make_shared<v1::Pad>(data, pads_begin, pads_end, pad_value_, ov_mode->second))};
}
}  // namespace

OutputVector translate_pad(const NodeContext& context) {
    num_inputs_check(context, 2, 4);
    auto data = context.get_input(0);
    auto paddings = get_input_concat_if_list(context, 1);
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

OutputVector translate_constant_pad_nd(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto data = context.get_input(0);
    auto paddings = get_input_concat_if_list(context, 1);
    Output<Node> pad_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    if (!context.input_is_none(2)) {
        pad_value = context.get_input(2);
    }
    return translate_pad_common(context, data, paddings, pad_value);
}

OutputVector translate_reflection_pad_nd(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto data = context.get_input(0);
    auto paddings = get_input_concat_if_list(context, 1);
    Output<Node> pad_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    return translate_pad_common(context, data, paddings, pad_value, "reflect");
}

OutputVector translate_replication_pad_nd(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto data = context.get_input(0);
    auto paddings = get_input_concat_if_list(context, 1);
    Output<Node> pad_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    return translate_pad_common(context, data, paddings, pad_value, "replicate");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov