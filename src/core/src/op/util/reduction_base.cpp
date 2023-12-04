// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/reduction_base.hpp"

#include "openvino/op/constant.hpp"
#include "reduce_shape_inference.hpp"
#include "validation_util.hpp"

ov::op::util::ReductionBase::ReductionBase() = default;

ov::op::util::ReductionBase::ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes}) {}

ov::PartialShape ov::op::util::ReductionBase::infer_reduction_output_shape(const bool keep_dims) {
    return reduce_shape_infer(this,
                              keep_dims,
                              std::vector<ov::PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)})
        .front();
}

bool ov::op::util::ReductionBase::reduction_axes_constant() const {
    return ov::is_type<op::v0::Constant>(input_value(1).get_node());
}

const ov::AxisSet ov::op::util::ReductionBase::get_reduction_axes() const {
    if (const auto& const_op = ov::util::get_constant_from_source(input_value(1))) {
        const auto const_data = const_op->cast_vector<int64_t>();
        const auto input_data_rank = get_input_partial_shape(0).rank();
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto normalized_axes = ov::normalize_axes(get_friendly_name(), const_data, input_data_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
        return {normalized_axes};
    } else {
        return {};
    }
}

void ov::op::util::ReductionBase::set_reduction_axes(const AxisSet& reduction_axes) {
    this->input(1).replace_source_output(
        op::v0::Constant::create(element::i64, ov::Shape{reduction_axes.size()}, reduction_axes.to_vector())
            ->output(0));
}
