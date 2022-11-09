// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset9;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
static void slice_pads_begin_end(const Output<Node>& paddings,
                                 shared_ptr<Node>& pads_begin,
                                 shared_ptr<Node>& pads_end) {
    // TODO: fix IR reader to accept padding of i32 type
    auto paddings_i64 = make_shared<Convert>(paddings, element::i64);
    auto axis = make_shared<Constant>(element::i32, Shape{}, 1);
    auto index_zero = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto index_one = make_shared<Constant>(element::i32, Shape{1}, 1);
    auto unsqueeze_pad_begin = make_shared<Gather>(paddings_i64, index_zero, axis);
    auto unsqueeze_pad_end = make_shared<Gather>(paddings_i64, index_one, axis);

    pads_begin = make_shared<Squeeze>(unsqueeze_pad_begin, axis);
    pads_end = make_shared<Squeeze>(unsqueeze_pad_end, axis);
}

static OutputVector translate_pad_base_op(const NodeContext& node,
                                          const Output<Node>& input,
                                          const Output<Node>& paddings,
                                          const Output<Node>& constant_value) {
    auto pad_mode = ov::op::PadMode::CONSTANT;

    // prepare pads_begin and pads_end for OpenVINO Pad
    shared_ptr<Node> pads_begin, pads_end;
    slice_pads_begin_end(paddings, pads_begin, pads_end);

    auto pad = make_shared<Pad>(input, pads_begin, pads_end, constant_value, pad_mode);
    set_node_name(node.get_name(), pad);
    return {pad};
}

OutputVector translate_pad_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Pad"});
    auto input = node.get_input(0);
    auto paddings = node.get_input(1);
    auto constant_value = make_shared<Constant>(input.get_element_type(), Shape{}, 0);

    return translate_pad_base_op(node, input, paddings, constant_value);
}

OutputVector translate_padv2_op(const NodeContext& node) {
    default_op_checks(node, 3, {"PadV2"});
    auto input = node.get_input(0);
    auto paddings = node.get_input(1);
    auto constant_value = node.get_input(2);

    return translate_pad_base_op(node, input, paddings, constant_value);
}

OutputVector translate_mirror_pad_op(const NodeContext& node) {
    default_op_checks(node, 2, {"MirrorPad"});
    auto input = node.get_input(0);
    auto paddings = node.get_input(1);

    // retrieve attributes
    auto mode = node.get_attribute<std::string>("mode");
    auto pad_mode = convert_padding_mode(node, mode);

    // prepare pads_begin and pads_end for OpenVINO Pad
    shared_ptr<Node> pads_begin, pads_end;
    slice_pads_begin_end(paddings, pads_begin, pads_end);

    auto pad = make_shared<Pad>(input, pads_begin, pads_end, pad_mode);
    set_node_name(node.get_name(), pad);
    return {pad};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
