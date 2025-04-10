// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_einsum_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Einsum"});

    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, op_type == "Einsum", "Internal error: incorrect usage of translate_einsum_op.");
    auto equation = node.get_attribute<string>("equation");
    int input_size = static_cast<int>(node.get_input_size());

    OutputVector inputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        inputs.push_back(node.get_input(input_ind));
    }

    auto einsum = make_shared<v7::Einsum>(inputs, equation);
    set_node_name(node.get_name(), einsum);
    return {einsum};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
