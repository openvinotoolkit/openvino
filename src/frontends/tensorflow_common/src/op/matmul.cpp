// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include "common_op_table.hpp"
#include "openvino/op/convert.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_mat_mul_op(const NodeContext& node) {
    default_op_checks(node, 2, {"MatMul"});

    auto a = node.get_input(0);
    auto b = node.get_input(1);
    auto transpose_a = node.get_attribute<bool>("transpose_a", false);
    auto transpose_b = node.get_attribute<bool>("transpose_b", false);

    auto res = make_shared<v0::MatMul>(a, b, transpose_a, transpose_b);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_batch_mat_mul_op(const NodeContext& node) {
    default_op_checks(node, 2, {"BatchMatMul", "BatchMatMulV2", "BATCH_MATMUL"});

    auto x = node.get_input(0);
    auto y = node.get_input(1);

    auto adj_x = node.get_attribute<bool>("adj_x", false);
    auto adj_y = node.get_attribute<bool>("adj_y", false);

    auto result = make_shared<v0::MatMul>(x, y, adj_x, adj_y);
    set_node_name(node.get_name(), result);
    return result->outputs();
}

OutputVector translate_batch_mat_mul_with_type_op(const NodeContext& node) {
    default_op_checks(node, 2, {"BatchMatMulV3"});

    auto x = node.get_input(0);
    auto y = node.get_input(1);

    auto input_type = x.get_element_type();

    auto adj_x = node.get_attribute<bool>("adj_x", false);
    auto adj_y = node.get_attribute<bool>("adj_y", false);
    auto t_out = node.get_attribute<ov::element::Type>("Tout", input_type);

    auto result = make_shared<v0::MatMul>(x, y, adj_x, adj_y)->output(0);

    if (t_out != input_type) {
        result = make_shared<v0::Convert>(result, t_out);
    }

    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
