// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/inverse.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_batch_matrix_inverse_op(const NodeContext& node) {
    default_op_checks(node, 1, {"BatchMatrixInverse"});
    auto input = node.get_input(0);
    bool adjoint = node.get_attribute<bool>("adjoint", false);
    auto result = std::make_shared<ov::op::v14::Inverse>(input, adjoint);
    set_node_name(node.get_name(), result);
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov