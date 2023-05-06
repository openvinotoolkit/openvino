// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/reduction_base.hpp"

#include "openvino/op/constant.hpp"
#include "reduce_shape_inference.hpp"

using namespace std;

ov::op::util::ReductionBase::ReductionBase() = default;

ov::op::util::ReductionBase::ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes}) {}

ov::PartialShape ov::op::util::ReductionBase::infer_reduction_output_shape(const bool keep_dims) {
    ov::PartialShape output_shape;
    reduce_shape_infer(this, keep_dims, get_input_partial_shape(0), output_shape);
    return output_shape;
}

bool ov::op::util::ReductionBase::reduction_axes_constant() const {
    return ov::is_type<op::v0::Constant>(input_value(1).get_node());
}

const ov::AxisSet ov::op::util::ReductionBase::get_reduction_axes() const {
    AxisSet axes;
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (const auto& const_op = get_constant_from_source(input_value(1))) {
        OPENVINO_SUPPRESS_DEPRECATED_END
        const auto const_data = const_op->cast_vector<int64_t>();
        const auto input_data_rank = get_input_partial_shape(0).rank();
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto normalized_axes = ov::normalize_axes(get_friendly_name(), const_data, input_data_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
        axes = AxisSet{normalized_axes};
    }
    return axes;
}

void ov::op::util::ReductionBase::set_reduction_axes(const AxisSet& reduction_axes) {
    this->input(1).replace_source_output(
        op::v0::Constant::create(element::i64, ov::Shape{reduction_axes.size()}, reduction_axes.to_vector())
            ->output(0));
}
