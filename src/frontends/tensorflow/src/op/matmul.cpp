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

OutputVector translate_mat_mul_op(const NodeContext& node) {
    auto a = node.get_input(0);
    auto b = node.get_input(1);
    auto transpose_a = node.get_attribute<bool>("transpose_a", false);
    auto transpose_b = node.get_attribute<bool>("transpose_b", false);

    auto res = make_shared<MatMul>(a, b, transpose_a, transpose_b);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
