// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_matrix_inverse_op(const NodeContext& node) {
    /*
    For matrix inverse translation, the goal is to convert the TensorFlow input tensor
    into an OpenVINO OutputVector that the Inverse operation can accept.
    */
    // a function that performs standard or default checks on a specific operation within the OpenVINO pipeline
    default_op_checks(node, 1, {"MatrixInverse"}); // copied from issue #22964
    // Retrieve the input tensor
    auto input = node.get_input(0);

    // Handle optional 'adjoint' attribute (default is False)
    bool adjoint = node.get_attribute<bool>("adjoint", false);


    // Create the OpenVINO equivalent operation (Inverse from opset-14)
    auto inverse_op = make_shared<ov::op::v14::Inverse>(input, false);
    set_node_name(node.get_name(), inverse_op);

    return {inverse_op};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov