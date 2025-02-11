// Copyright (C) 2018-2025 Intel Corporation
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

using namespace ov::op;

namespace {
Output<Node> create_padding(ov::pass::NodeRegistry& rg,
                            const Output<Node>& input_rank,
                            const Output<Node>& padding,
                            const Output<Node>& start_id,
                            const Output<Node>& end_id) {
    // PyTorch paddings represented as [N_pad_begins, N_pad_ends, N-1_pad_begins, N-1_pad_ends, ... ]
    // if len of paddings not equal to input rank * 2, zero padding added to first rank - N  dimensions
    // OV expects paddings separated on begins and ends for each dimension from first to last
    auto minus_two = v0::Constant::create(element::i32, Shape{}, {-2});
    auto zero = v0::Constant::create(element::i32, Shape{}, {0});
    auto pad_id_range = rg.make<v4::Range>(start_id, end_id, minus_two, element::i32);
    auto pads = rg.make<v8::Gather>(padding, pad_id_range, zero);
    // add left side zero padding for difference between padding size and input rank
    auto pads_short_len = rg.make<v3::ShapeOf>(pads, element::i32);
    auto pads_diff = rg.make<v1::Subtract>(input_rank, pads_short_len);
    auto pads_remaining = rg.make<v3::Broadcast>(zero, pads_diff);
    auto pads_remaining_c = rg.make<v1::ConvertLike>(pads_remaining, pads);
    auto pads_full = rg.make<v0::Concat>(OutputVector{pads_remaining_c, pads}, 0);
    return pads_full;
}

const std::unordered_map<std::string, PadMode> PAD_MODES = {{"constant", PadMode::CONSTANT},
                                                            {"reflect", PadMode::REFLECT},
                                                            {"replicate", PadMode::EDGE}};

};  // namespace

PrimListConstructPadReplacer::PrimListConstructPadReplacer() {
    // transformation for case aten::pad + prim::ListConstruct as paddings
    const auto& pad_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        Output<Node> input_node;
        Output<Node> padding;
        Output<Node> pad_value;
        std::string mode;
        std::shared_ptr<ov::op::util::FrameworkNode> pad_op;
        if ((pad_op = cast_fw_node(m.get_match_root(), "aten::pad"))) {
            mode = "constant";
            input_node = pad_op->input_value(0);
            padding = pad_op->input_value(1);
            const auto& mode_node = pad_op->input_value(2).get_node_shared_ptr();
            if (const auto& fw_node_mode = cast_fw_node(mode_node, "prim::Constant")) {
                const auto& attrs = fw_node_mode->get_attrs();
                if (attrs.find("string_value") != attrs.end()) {
                    mode = attrs.at("string_value");
                }
            }
            pad_value = pad_op->input_value(3);
            if (const auto& fw_node_mode = cast_fw_node(pad_value.get_node_shared_ptr(), "prim::Constant")) {
                const auto& attrs = fw_node_mode->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    pad_value = v0::Constant::create(element::f32, Shape{}, {0});
                }
            }
        } else if ((pad_op = cast_fw_node(m.get_match_root(), "aten::reflection_pad2d"))) {
            mode = "reflect";
            input_node = pad_op->input_value(0);
            padding = pad_op->input_value(1);
            // Pad value is used only for constant pad, fill with 0 as placeholder.
            pad_value = v0::Constant::create(element::f32, Shape{}, {0});
        } else {
            return false;
        }
        ov::pass::NodeRegistry rg;
        auto minus_two = v0::Constant::create(element::i32, Shape{}, {-2});
        auto minus_one = v0::Constant::create(element::i32, Shape{}, {-1});
        auto zero = v0::Constant::create(element::i32, Shape{}, {0});
        // for case. when padding is list of scalars, concatenate them into one tensor
        auto pad_values = concat_list_construct(padding);
        auto zero_f = v0::Constant::create(element::f32, Shape{}, {0});
        auto input_shape = rg.make<v3::ShapeOf>(input_node, element::i32);
        auto input_rank = rg.make<v3::ShapeOf>(input_shape, element::i32);
        auto pad_size_1d = rg.make<v3::ShapeOf>(pad_values, element::i32);
        auto pad_size = rg.make<v0::Squeeze>(pad_size_1d, zero);
        // get pad_begins and pad_ends indexes starting for end of paddings
        auto start_pad_begins = rg.make<v1::Add>(pad_size, minus_two);
        auto start_pad_ends = rg.make<v1::Add>(pad_size, minus_one);
        auto pad_begins_full = create_padding(rg, input_rank, pad_values, start_pad_begins, minus_one);
        auto pad_ends_full = create_padding(rg, input_rank, pad_values, start_pad_ends, zero);
        if (mode == "constant") {
            if (const auto& fw_node_value = cast_fw_node(pad_value.get_node_shared_ptr(), "prim::Constant")) {
                const auto& attrs = fw_node_value->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    pad_value = zero_f;
                }
            }
            pad_value = rg.make<v1::ConvertLike>(pad_value, input_node);
        }
        if (PAD_MODES.find(mode) == PAD_MODES.end()) {
            add_exception_to_fw_node(pad_op, "Unsupported mode: " + mode + "for aten::pad");
            return false;
        }
        auto pad_mode = PAD_MODES.at(mode);
        auto pad = rg.make<v1::Pad>(input_node, pad_begins_full, pad_ends_full, pad_value, pad_mode);
        replace_node(pad_op, pad);
        copy_runtime_info_and_name(pad_op, rg.get());
        pad->set_friendly_name(pad_op->get_friendly_name());
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
