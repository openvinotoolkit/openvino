// Copyright (C) 2018-2023 Intel Corporation
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
    default_op_checks(node, 1, {"Softmax"});
    auto logits = node.get_input(0);
    auto softmax = make_shared<opset8::Softmax>(logits, -1);
    set_node_name(node.get_name(), softmax);
    return {softmax};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
