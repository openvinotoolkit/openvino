// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/grouped_matmul_compressed.hpp"

#include "grouped_matmul_shape_inference.hpp"

namespace ov::op::internal {

GroupedMatMulCompressed::GroupedMatMulCompressed(const ov::Output<Node>& mat_a,
                                                 const ov::Output<Node>& mat_b,
                                                 const ov::Output<Node>& offsets,
                                                 const ov::Output<Node>& decompression_scale)
    : ov::op::v17::GroupedMatMul() {
    set_arguments(ov::OutputVector{mat_a, mat_b, offsets, decompression_scale});
    validate_and_infer_types();
}

GroupedMatMulCompressed::GroupedMatMulCompressed(const ov::Output<Node>& mat_a,
                                                 const ov::Output<Node>& mat_b,
                                                 const ov::Output<Node>& offsets,
                                                 const ov::Output<Node>& decompression_scale,
                                                 const ov::Output<Node>& decompression_zero_point)
    : ov::op::v17::GroupedMatMul() {
    set_arguments(ov::OutputVector{mat_a, mat_b, offsets, decompression_scale, decompression_zero_point});
    validate_and_infer_types();
}

std::shared_ptr<GroupedMatMulCompressed> GroupedMatMulCompressed::make_3d(const ov::Output<Node>& mat_a,
                                                                          const ov::Output<Node>& mat_b,
                                                                          const ov::Output<Node>& decompression_scale) {
    auto op = std::make_shared<GroupedMatMulCompressed>();
    op->set_arguments(ov::OutputVector{mat_a, mat_b, decompression_scale});
    op->validate_and_infer_types();
    return op;
}

std::shared_ptr<GroupedMatMulCompressed> GroupedMatMulCompressed::make_3d(
    const ov::Output<Node>& mat_a,
    const ov::Output<Node>& mat_b,
    const ov::Output<Node>& decompression_scale,
    const ov::Output<Node>& decompression_zero_point) {
    auto op = std::make_shared<GroupedMatMulCompressed>();
    op->set_arguments(ov::OutputVector{mat_a, mat_b, decompression_scale, decompression_zero_point});
    op->validate_and_infer_types();
    return op;
}

bool GroupedMatMulCompressed::has_offsets() const {
    // The 2D x 3D form always carries an offsets input; the 3D x 3D form does not.
    // Detection uses the rank of mat_a since scale/zp are always attached after the
    // base GroupedMatMul inputs.
    const auto& a_rank = get_input_partial_shape(0).rank();
    return a_rank.is_static() && a_rank.get_length() == 2;
}

void GroupedMatMulCompressed::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size >= 3,
                          "GroupedMatMulCompressed expects at least 3 inputs, got: ",
                          input_size);

    std::vector<ov::PartialShape> input_shapes{get_input_partial_shape(0), get_input_partial_shape(1)};
    if (has_offsets()) {
        input_shapes.push_back(get_input_partial_shape(2));
    }
    const auto out_shapes = ov::op::v17::shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), out_shapes[0]);
}

std::shared_ptr<ov::Node> GroupedMatMulCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    auto op = std::make_shared<GroupedMatMulCompressed>();
    op->set_arguments(new_args);
    op->validate_and_infer_types();
    return op;
}

}  // namespace ov::op::internal
