// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslatePadOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_paddings_op = node.get_ng_input(1);
    Output<Node> pad_val_op;

    // Set inputs and pad_val_op
    auto op_type = node.get_op_type();
    if (op_type == "Pad" || op_type == "MirrorPad") {
        pad_val_op =
            ConstructNgNode<Constant>(node.get_name(), ng_input.get_element_type(), Shape(), std::vector<int>({0}));
    } else if (op_type == "PadV2") {
        pad_val_op = node.get_ng_input(2);
    } else {
        throw errors::InvalidArgument("Incorrect TF Pad OpType: " + node.get_op_type());
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
            throw errors::InvalidArgument(pad_mode_str + " is not an allowed padding mode.");
        }
    }

    // Set pads_begin & pads_end (from the pad_val_op)
    std::vector<int64_t> paddings;
    GetStaticInputVector(node, 1, &paddings);
    if (paddings.size() % 2 != 0) {
        throw errors::InvalidArgument("Constant node for paddings does not have an even number of "
                                      "elements");
    }
    std::vector<int64_t> pad_begin(paddings.size() / 2);
    std::vector<int64_t> pad_end(paddings.size() / 2);
    for (size_t i = 0; i < paddings.size() / 2; i++) {
        pad_begin[i] = paddings[2 * i];
        pad_end[i] = paddings[2 * i + 1];
    }
    auto pads_begin_node = ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{pad_begin.size()}, pad_begin);
    auto pads_end_node = ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{pad_end.size()}, pad_end);

    // Create final Op
    auto result_pad_op =
        ConstructNgNode<Pad>(node.get_name(), ng_input, pads_begin_node, pads_end_node, pad_val_op, pad_mode);

    return {result_pad_op};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
