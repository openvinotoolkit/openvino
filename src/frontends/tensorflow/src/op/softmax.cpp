// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset7.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_softmax_op(const NodeContext& node) {
    auto ng_inp = node.get_input(0);
    // todo: switch to opset8::Softmax when is ready and delete Dynamic rank limitation
    TENSORFLOW_OP_VALIDATION(node, ng_inp.get_partial_shape().rank().is_static(), "Dynamic rank is not supported.");
    size_t axis = ng_inp.get_partial_shape().rank().get_length() - 1;
    auto res = make_shared<opset7::Softmax>(ng_inp, axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov