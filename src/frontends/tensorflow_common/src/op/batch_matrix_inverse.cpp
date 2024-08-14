// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/inverse.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_batch_matrix_inverse_op(const NodeContext& node) {
    default_op_checks(node, 1, {"BatchMatrixInverse"});
    auto input = node.get_input(0);
    const auto input_shape = input.get_partial_shape();
    size_t rank = input_shape.rank().get_length();
    TENSORFLOW_OP_VALIDATION(node, rank >= 3, "BatchMatrixInverse input must have at least 3 dimensions");

    bool has_zero_dim = false;
    if (input_shape.is_static()) {
        has_zero_dim = std::any_of(input_shape.begin(), input_shape.end() - 2, [](const Dimension& dim) {
            return dim.is_static() && dim.get_length() == 0;
        });
    }
    std::shared_ptr<Node> result;
    if (has_zero_dim) {
        result = input.get_node_shared_ptr();
    } else {
        bool adjoint = node.get_attribute<bool>("adjoint", false);
        result = std::make_shared<ov::op::v14::Inverse>(input, adjoint);
    }

    set_node_name(node.get_name(), result);
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov