// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prim_list_construct_pad.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::opset10;
namespace {
std::shared_ptr<Node> create_padding(std::shared_ptr<Node> input_rank,
                                     std::shared_ptr<Node> padding,
                                     std::shared_ptr<Node> start_id,
                                     std::shared_ptr<Node> end_id) {
    // PyTorch paddings represented as [N_pad_begins, N_pad_ends, N-1_pad_begins, N-1_pad_ends, ... ]
    // if len of paddings not equal to input rank * 2, zero padding added to first rank - N  dimensions
    // OV expects paddings separated on begins and ends for each dimension from first to last
    auto minus_two = Constant::create(element::i64, Shape{}, {-2});
    auto zero = Constant::create(element::i64, Shape{}, {0});
    auto pad_id_range = std::make_shared<Range>(start_id, end_id, minus_two, element::i64);
    auto pads = std::make_shared<Gather>(padding, pad_id_range, zero);
    // add left side zero padding for difference between padding size and input rank
    auto pads_short_len = std::make_shared<ShapeOf>(pads);
    auto pads_diff = std::make_shared<Subtract>(input_rank, pads_short_len);
    auto pads_remaining = std::make_shared<Broadcast>(zero, pads_diff);
    auto pads_remaining_c = std::make_shared<ConvertLike>(pads_remaining, pads);
    auto pads_full = std::make_shared<Concat>(OutputVector{pads_remaining_c, pads}, 0);
    return pads_full;
}

};  // namespace

PrimListConstructPadReplacer::PrimListConstructPadReplacer() {
    // transformation for case aten::pad + prim::ListConstruct as paddings
    auto pad_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto pad_op = cast_fw_node(m.get_match_root(), "aten::pad");
        if (!pad_op) {
            return false;
        }
        auto minus_two = Constant::create(element::i64, Shape{}, {-2});
        auto minus_one = Constant::create(element::i64, Shape{}, {-1});
        auto zero = Constant::create(element::i64, Shape{}, {0});
        auto input_node = pad_op->input_value(0).get_node_shared_ptr();
        auto padding = pad_op->input_value(1).get_node_shared_ptr();
        // for case. when padding is list of scalars, concatenate them into one tensor
        auto pad_values = concat_list_construct(padding);
        std::string mode = "constant";
        auto zero_f = Constant::create(element::f32, Shape{}, {0});
        auto input_shape = std::make_shared<ShapeOf>(input_node);
        auto input_rank = std::make_shared<ShapeOf>(input_shape);
        auto pad_size_1d = std::make_shared<ShapeOf>(pad_values);
        auto pad_size = std::make_shared<Squeeze>(pad_size_1d, zero);
        // get pad_begins and pad_ends indexes starting for end of paddings
        auto start_pad_begins = std::make_shared<Add>(pad_size, minus_two);
        auto start_pad_ends = std::make_shared<Add>(pad_size, minus_one);
        auto pad_begins_full = create_padding(input_rank, pad_values, start_pad_begins, minus_one);
        auto pad_ends_full = create_padding(input_rank, pad_values, start_pad_ends, zero);
        std::shared_ptr<Node> pad;
        // PtFrameworkNode decoder has information about python type constnats like None and string values
        auto pt_node = std::dynamic_pointer_cast<PtFrameworkNode>(pad_op);
        auto decoder = pt_node->get_decoder();
        if (!decoder->input_is_none(2)) {
            auto mode_const = pad_op->input_value(2).get_node_shared_ptr();
            mode_const = cast_fw_node(mode_const, "prim::Constant");
            auto pt_mode = std::dynamic_pointer_cast<PtFrameworkNode>(mode_const);
            mode = pt_mode->get_decoder()->as_string();
        }
        if (mode == "constant") {
            if (!decoder->input_is_none(3)) {
                auto pad_value = pad_op->input_value(3).get_node_shared_ptr();
                pad = std::make_shared<Pad>(input_node,
                                            pad_begins_full,
                                            pad_ends_full,
                                            pad_value,
                                            ov::op::PadMode::CONSTANT);
            } else {
                pad = std::make_shared<Pad>(input_node,
                                            pad_begins_full,
                                            pad_ends_full,
                                            zero_f,
                                            ov::op::PadMode::CONSTANT);
            }
        } else if (mode == "reflect") {
            pad = std::make_shared<Pad>(input_node, pad_begins_full, pad_ends_full, zero_f, ov::op::PadMode::REFLECT);
        } else if (mode == "replicate") {
            pad = std::make_shared<Pad>(input_node, pad_begins_full, pad_ends_full, zero_f, ov::op::PadMode::EDGE);
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "aten::pad conversion doesn't support [ " + mode + " ] padding mode");
        }
        replace_node(pad_op, pad);
        copy_runtime_info({pad_op,
                           input_node,
                           padding,
                           pad_op->input_value(2).get_node_shared_ptr(),
                           pad_op->input_value(3).get_node_shared_ptr()},
                          pad);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pad_op,
                                                          "ov::frontend::pytorch::pass::PrimListConstructPadReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov