// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/arithmetic_reduction.hpp"

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::ArithmeticReduction);

ov::op::util::ArithmeticReduction::ArithmeticReduction() = default;

ov::op::util::ArithmeticReduction::ArithmeticReduction(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ReductionBase(arg, reduction_axes) {}

bool ov::op::util::ArithmeticReduction::reduction_axes_constant() const {
    return ov::is_type<ngraph::op::Constant>(input_value(1).get_node());
}

const ov::AxisSet ov::op::util::ArithmeticReduction::get_reduction_axes() const {
    AxisSet axes;
    if (const auto& const_op = get_constant_from_source(input_value(1))) {
        const auto const_data = const_op->cast_vector<int64_t>();
        const auto input_data_rank = get_input_partial_shape(0).rank();
        const auto normalized_axes = ngraph::normalize_axes(get_friendly_name(), const_data, input_data_rank);
        axes = AxisSet{normalized_axes};
    }
    return axes;
}

void ov::op::util::ArithmeticReduction::set_reduction_axes(const AxisSet& reduction_axes) {
    this->input(1).replace_source_output(
        ngraph::op::Constant::create(element::i64, ov::Shape{reduction_axes.size()}, reduction_axes.to_vector())
            ->output(0));
}

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
