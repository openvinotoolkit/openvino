// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_gather_matmul.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "matmul_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/op.hpp"
#include "transformations/itt.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

BatchGatherMatmul::BatchGatherMatmul(const ov::Output<Node>& A,
                                     const ov::Output<Node>& B,
                                     const ov::Output<Node>& indices,
                                     const ov::Output<Node>& bias)
    : Op({A, B, indices, bias}) {
    validate_and_infer_types();
}

BatchGatherMatmul::BatchGatherMatmul(const ov::Output<Node>& A,
                                     const ov::Output<Node>& B,
                                     const ov::Output<Node>& indices)
    : BatchGatherMatmul(A, B, indices, std::make_shared<ov::op::v0::Constant>(element::dynamic, Shape{0})) {}

std::shared_ptr<ov::Node> BatchGatherMatmul::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(GroupGatherMatmul_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::BatchGatherMatmul>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3));
}

void BatchGatherMatmul::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(GroupGatherMatmul_validate_and_infer_types);
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size >= 4,
                          "Number of inputs is incorrect. Current value is: ",
                          input_size,
                          ", expected at least 4.");

    // Check input B is on constant path
    NODE_VALIDATION_CHECK(this,
                          ov::op::util::is_on_path<ov::op::v0::Constant>(input_value(1)),
                          "Input B must be on constant path.");

    const auto& a_shape = get_input_partial_shape(0);
    const auto& b_shape = get_input_partial_shape(1);
    const auto& indices_shape = get_input_partial_shape(2);
    const auto& bias_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this, a_shape.rank().is_static(), "Input A rank must be static.");
    NODE_VALIDATION_CHECK(this, b_shape.rank().is_static(), "Input B rank must be static.");
    NODE_VALIDATION_CHECK(this, indices_shape.rank().is_static(), "Input indices rank must be static.");
    NODE_VALIDATION_CHECK(this, bias_shape.rank().is_static(), "Input bias rank must be static.");

    const size_t a_rank = a_shape.size();
    const size_t b_rank = b_shape.size();
    const size_t indices_rank = indices_shape.size();
    const size_t bias_rank = bias_shape.is_dynamic() ? 0 : bias_shape.size();

    NODE_VALIDATION_CHECK(this, a_rank == 3, "Input A rank must be exactly 3D. Got: ", a_rank, "D instead.");
    NODE_VALIDATION_CHECK(this, b_rank == 3, "Input B rank must be exactly 3D. Got: ", b_rank, "D instead.");
    NODE_VALIDATION_CHECK(this,
                          indices_rank == 2,
                          "Input indices rank must be exactly 2D. Got: ",
                          indices_rank,
                          "D instead.");
    NODE_VALIDATION_CHECK(this,
                          (bias_rank == 1 || bias_rank == 0 || bias_rank == a_rank),
                          "Input bias rank must be either 1D, scalar or same as Input A rank. Got: ",
                          bias_rank,
                          "D instead.");

    if (indices_shape[1].is_static()) {
        if (a_shape[0].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  a_shape[0] == 1 || a_shape[0] == indices_shape[1],
                                  "The first dimension of input A must be equal to number of activated experts. Got: ",
                                  a_shape[0],
                                  " and ",
                                  indices_shape[1],
                                  " instead.");
        }
        if (b_shape[0].is_static()) {
            NODE_VALIDATION_CHECK(
                this,
                indices_shape[1].get_length() <= b_shape[0].get_length(),
                "The second dimension of input indices must be less or equal to the first dimension of input B. "
                "Got: ",
                indices_shape[1],
                " and ",
                b_shape[0],
                " instead.");
        }
    }

    if (indices_shape[0].is_static() && a_shape[1].is_static()) {
        NODE_VALIDATION_CHECK(this,
                              indices_shape[0] == a_shape[1],
                              "The first dimension of input indices must be equal to the second dimension of input A. "
                              "Got: ",
                              indices_shape[0],
                              " and ",
                              a_shape[1],
                              " instead.");
    }

    ov::op::v0::MatMul op;
    op.set_transpose_a(transp_a);
    op.set_transpose_b(transp_b);

    ov::PartialShape matmul_shape_a = {a_shape[1], a_shape[2]};
    ov::PartialShape matmul_shape_b = {b_shape[1], b_shape[2]};

    auto out_matmul_shape =
        (ov::op::v0::shape_infer(&op, std::vector<ov::PartialShape>{matmul_shape_a, matmul_shape_b})).front();

    ov::PartialShape output_shape = {indices_shape[1], out_matmul_shape[0], out_matmul_shape[1]};

    set_output_type(0, get_input_element_type(0), output_shape);
}

}  // namespace ov::intel_cpu