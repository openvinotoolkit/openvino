// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/matmul.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_sparse_tensor_dense_mat_mul_op(const NodeContext& node) {
    default_op_checks(node, 4, {"SparseTensorDenseMatMul"});
    auto a_indices = node.get_input(0);
    auto a_values = node.get_input(1);
    auto a_shape = node.get_input(2);
    auto b = node.get_input(3);
    auto adjoint_a = node.get_attribute<bool>("adjoint_a", false);
    auto adjoint_b = node.get_attribute<bool>("adjoint_b", false);

    auto a = create_dense_tensor(a_indices, a_shape, a_values);
    auto res = make_shared<v0::MatMul>(a, b, adjoint_a, adjoint_b);
    set_node_name(node.get_name(), res);
    return {res};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
