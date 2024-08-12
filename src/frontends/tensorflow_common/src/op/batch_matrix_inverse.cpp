// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/inverse.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_batch_matrix_inverse_op(const NodeContext& node) {
    const auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, op_type == "BatchMatrixInverse", "Internal error: incorrect usage of translate_batch_matrix_inverse_op.");

    const auto input = node.get_input(0);
    const bool adjoint = node.get_attribute<bool>("adjoint", false);

    const auto input_shape = input.get_partial_shape();
    TENSORFLOW_OP_VALIDATION(node, input_shape.rank().is_static(), "BatchMatrixInverse input must have static rank");
    const auto rank = input_shape.rank().get_length();
    TENSORFLOW_OP_VALIDATION(node, rank >= 3, "BatchMatrixInverse input must have at least 3 dimensions");

    std::vector<int64_t> combined_shape = {-1};
    for (int i = rank - 2; i < rank; ++i ) {
        combined_shape.push_back(input_shape[i].get_length());
    }

    auto reshape_to_2d = std::make_shared<ov::op::v1::Reshape>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, {combined_shape.size()}, combined_shape),
        false
    );
    auto inverse_2d = std::make_shared<ov::op::v14::Inverse>(reshape_to_2d, false);
    auto original_shape = ov::op::v0::Constant::create(ov::element::i64, {static_cast<size_t>(rank)}, input_shape.get_shape());
    auto reshape_back = std::make_shared<ov::op::v1::Reshape>(inverse_2d, original_shape, false);

    set_node_name(node.get_name(), reshape_back);
    return {reshape_back};
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov