// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_softmax_op(const NodeContext& node) {
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() > 0, "Softmax must have at least one input.");
    auto input = node.get_input(0);
    auto res = make_shared<opset8::Softmax>(input, -1);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov