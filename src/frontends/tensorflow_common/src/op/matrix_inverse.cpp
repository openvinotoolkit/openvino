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
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, op_type == "MatrixInverse", "Internal error: incorrect usage of translate_matrix_inverse_op.");
    auto equation = node.get_attribute<std::string>("equation"); 

    OutputVector inputs;
    for (size_t input_ind = 0; input_ind < node.get_input_size(); ++input_ind) {
        inputs.push_back(node.get_input(input_ind));
    }

    // Create the OpenVINO equivalent operation (Inverse from opset-14)
    auto inverse_op = make_shared<ov::op::v14::Inverse>(inputs[0], false);
    set_node_name(node.get_name(), inverse_op);
    return {inverse_op};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov