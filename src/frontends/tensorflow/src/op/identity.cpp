// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_identity_op(const NodeContext& node) {
    auto input = node.get_input(0);

    // since the input node can have several outputs, and identity have only one input,
    // we cannot use set_node_name(..) helper, we have to set names for output connected
    // to this identity only.
    // Node_1 -> Node_2
    //        -(identity name) -> Identity
    set_out_name(node.get_name(), input);
    set_out_name(node.get_name() + ":" + "0", input);
    return {input};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov