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
#include "openvino/op/squeeze.hpp"
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

namespace {
std::shared_ptr<Node> create_padding(std::shared_ptr<Node> input_rank,
                                     std::shared_ptr<Node> padding,
                                     std::shared_ptr<Node> start_id,
                                     std::shared_ptr<Node> end_id) {
    // PyTorch paddings represented as [N_pad_begins, N_pad_ends, N-1_pad_begins, N-1_pad_ends, ... ]
    // if len of paddings not equal to input rank * 2, zero padding added to first rank - N  dimensions
    // OV expects paddings separated on begins and ends for each dimension from first to last
    auto minus_two = ov::op::v0::Constant::create(element::i32, Shape{}, {-2});
    auto zero = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    auto pad_id_range = std::make_shared<ov::op::v4::Range>(start_id, end_id, minus_two, element::i32);
    auto pads = std::make_shared<ov::op::v8::Gather>(padding, pad_id_range, zero);
    // add left side zero padding for difference between padding size and input rank
    auto pads_short_len = std::make_shared<ov::op::v3::ShapeOf>(pads, element::i32);
    auto pads_diff = std::make_shared<ov::op::v1::Subtract>(input_rank, pads_short_len);
    auto pads_remaining = std::make_shared<ov::op::v3::Broadcast>(zero, pads_diff);
    auto pads_remaining_c = std::make_shared<ov::op::v1::ConvertLike>(pads_remaining, pads);
    auto pads_full = std::make_shared<ov::op::v0::Concat>(OutputVector{pads_remaining_c, pads}, 0);
    return pads_full;
}

const std::unordered_map<std::string, ov::op::PadMode> PAD_MODES = {{"constant", ov::op::PadMode::CONSTANT},
                                                                    {"reflect", ov::op::PadMode::REFLECT},
                                                                    {"replicate", ov::op::PadMode::EDGE}};

};  // namespace

PrimListConstructPadReplacer::PrimListConstructPadReplacer() {
    // transformation for case aten::pad + prim::ListConstruct as paddings
    auto pad_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto pad_op = cast_fw_node(m.get_match_root(), "aten::pad");
        if (!pad_op) {
            return false;
        }
        auto minus_two = ov::op::v0::Constant::create(element::i32, Shape{}, {-2});
        auto minus_one = ov::op::v0::Constant::create(element::i32, Shape{}, {-1});
        auto zero = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
        auto input_node = pad_op->input_value(0).get_node_shared_ptr();
        auto padding = pad_op->input_value(1).get_node_shared_ptr();
        // for case. when padding is list of scalars, concatenate them into one tensor
        auto pad_values = concat_list_construct(padding);
        std::string mode = "constant";
        auto zero_f = ov::op::v0::Constant::create(element::f32, Shape{}, {0});
        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_node, element::i32);
        auto input_rank = std::make_shared<ov::op::v3::ShapeOf>(input_shape, element::i32);
        auto pad_size_1d = std::make_shared<ov::op::v3::ShapeOf>(pad_values, element::i32);
        auto pad_size = std::make_shared<ov::op::v0::Squeeze>(pad_size_1d, zero);
        // get pad_begins and pad_ends indexes starting for end of paddings
        auto start_pad_begins = std::make_shared<ov::op::v1::Add>(pad_size, minus_two);
        auto start_pad_ends = std::make_shared<ov::op::v1::Add>(pad_size, minus_one);
        auto pad_begins_full = create_padding(input_rank, pad_values, start_pad_begins, minus_one);
        auto pad_ends_full = create_padding(input_rank, pad_values, start_pad_ends, zero);
        auto mode_const = pad_op->input_value(2).get_node_shared_ptr();
        auto pad_value = pad_op->input_value(3).get_node_shared_ptr();
        if (const auto& fw_node_mode = cast_fw_node(mode_const, "prim::Constant")) {
            const auto& attrs = fw_node_mode->get_attrs();
            if (attrs.find("string_value") != attrs.end()) {
                mode = attrs.at("string_value");
            }
        }
        if (mode == "constant") {
            if (const auto& fw_node_value = cast_fw_node(pad_value, "prim::Constant")) {
                const auto& attrs = fw_node_value->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    pad_value = zero_f;
                }
            }
            pad_value = std::make_shared<ov::op::v1::ConvertLike>(pad_value, input_node);
        }
        FRONT_END_OP_CONVERSION_CHECK(PAD_MODES.find(mode) != PAD_MODES.end(),
                                      "Unsupported mode: ",
                                      mode,
                                      "for aten::pad");
        auto pad_mode = PAD_MODES.at(mode);
        auto pad = std::make_shared<ov::op::v1::Pad>(input_node, pad_begins_full, pad_ends_full, pad_value, pad_mode);
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