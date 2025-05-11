// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/logical_reduction.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"

namespace ov {

op::util::LogicalReduction::LogicalReduction() = default;

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg, const AxisSet& reduction_axes)
    : ReductionBase(
          arg,
          ov::op::v0::Constant::create(element::i64, ov::Shape{reduction_axes.size()}, reduction_axes.to_vector())
              ->output(0)) {}

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ReductionBase(arg, reduction_axes) {}

void op::util::LogicalReduction::validate_and_infer_types() {
    OV_OP_SCOPE(util_LogicalReduction_validate_and_infer_types);

    const element::Type& data_et = get_input_element_type(0);
    const PartialShape& axes_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, data_et.compatible(element::boolean), "Element type of data input must be boolean.");

    const Rank axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    PartialShape result_shape = infer_reduction_output_shape(false);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, data_et, result_shape);
}

}  // namespace ov
