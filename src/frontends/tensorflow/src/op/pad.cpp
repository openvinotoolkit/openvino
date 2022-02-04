// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_pad_op(const NodeContext& node) {
    auto ng_input = node.get_input(0), ng_paddings_op = node.get_input(1);
    Output<Node> pad_val_op;

    // Set inputs and pad_val_op
    auto op_type = node.get_op_type();
    if (op_type == "Pad" || op_type == "MirrorPad") {
        pad_val_op = make_shared<Constant>(ng_input.get_element_type(), Shape(), std::vector<int>({0}));
    } else if (op_type == "PadV2") {
        pad_val_op = node.get_input(2);
    } else {
        TENSORFLOW_OP_VALIDATION(node, false, "Incorrect TF Pad OpType: " + node.get_op_type());
    }

    // Set pad_mode
    auto pad_mode = ov::op::PadMode::CONSTANT;
    if (op_type == "MirrorPad") {
        auto pad_mode_str = node.get_attribute<std::string>("mode");
        if (pad_mode_str == "REFLECT") {
            pad_mode = ov::op::PadMode::REFLECT;
        } else if (pad_mode_str == "SYMMETRIC") {
            pad_mode = ov::op::PadMode::SYMMETRIC;
        } else {
            TENSORFLOW_OP_VALIDATION(node, false, pad_mode_str + " is not an allowed padding mode.");
        }
    }

    // Set pads_begin & pads_end (from the pad_val_op)
    std::vector<int64_t> paddings;
    get_const_input(node, 1, &paddings);
    if (paddings.size() % 2 != 0) {
        TENSORFLOW_OP_VALIDATION(node,
                                 false,
                                 "Constant node for paddings does not have an even number of "
                                 "elements");
    }
    std::vector<int64_t> pad_begin(paddings.size() / 2);
    std::vector<int64_t> pad_end(paddings.size() / 2);
    for (size_t i = 0; i < paddings.size() / 2; i++) {
        pad_begin[i] = paddings[2 * i];
        pad_end[i] = paddings[2 * i + 1];
    }
    auto pads_begin_node = make_shared<Constant>(element::i64, Shape{pad_begin.size()}, pad_begin);
    auto pads_end_node = make_shared<Constant>(element::i64, Shape{pad_end.size()}, pad_end);

    // Create final Op
    auto res = make_shared<Pad>(ng_input, pads_begin_node, pads_end_node, pad_val_op, pad_mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov