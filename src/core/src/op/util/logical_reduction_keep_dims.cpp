// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/logical_reduction_keep_dims.hpp"

#include "itt.hpp"

ov::op::util::LogicalReductionKeepDims::LogicalReductionKeepDims(const ov::Output<ov::Node>& arg,
                                                                 const ov::Output<ov::Node>& reduction_axes,
                                                                 const bool keep_dims)
    : LogicalReduction(arg, reduction_axes),
      m_keep_dims{keep_dims} {}

bool ov::op::util::LogicalReductionKeepDims::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_util_LogicalReductionKeepDims_visit_attributes);
    visitor.on_attribute("keep_dims", m_keep_dims);
    return true;
}

void ov::op::util::LogicalReductionKeepDims::validate_and_infer_types() {
    OV_OP_SCOPE(v0_util_LogicalReductionKeepDims_validate_and_infer_types);

    const element::Type& data_et = get_input_element_type(0);
    const element::Type& axes_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this, data_et.compatible(element::boolean), "Element type of data input must be boolean.");

    NODE_VALIDATION_CHECK(this,
                          axes_et.is_integral_number(),
                          "Element type of axes input must be integer. Got: ",
                          axes_et);

    PartialShape result_shape = infer_reduction_output_shape(m_keep_dims);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, data_et, result_shape);
}
