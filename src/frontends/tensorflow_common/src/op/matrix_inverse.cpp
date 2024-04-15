// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/inverse.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_matrix_inverse_op(const NodeContext& node) {
    default_op_checks(node, 1, {"MatrixInverse"});
    // retrieve the input tensor
    auto input = node.get_input(0);

    // handle optional 'adjoint' attribute (default is false)
    bool adjoint = node.get_attribute<bool>("adjoint", false);
    TENSORFLOW_OP_VALIDATION(
        node,
        !adjoint,
        "[TensorFlow Frontend] internal error: MatrixInverse is supported only for adjoint equal to false");

    auto inverse_op = make_shared<ov::op::v14::Inverse>(input, adjoint);
    set_node_name(node.get_name(), inverse_op);

    return {inverse_op};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov