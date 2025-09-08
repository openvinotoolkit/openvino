// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/arithmetic_reduction.hpp"

#include "itt.hpp"

ov::op::util::ArithmeticReduction::ArithmeticReduction() = default;

ov::op::util::ArithmeticReduction::ArithmeticReduction(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ReductionBase(arg, reduction_axes) {}

void ov::op::util::ArithmeticReduction::validate_and_infer_types() {
    OV_OP_SCOPE(util_ArithmeticReduction_validate_and_infer_types);

    const PartialShape& axes_shape = get_input_partial_shape(1);
    const Rank axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    PartialShape result_shape = infer_reduction_output_shape(false);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, get_input_element_type(0), result_shape);
}
