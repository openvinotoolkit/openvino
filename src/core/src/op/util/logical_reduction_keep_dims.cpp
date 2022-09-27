// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/logical_reduction_keep_dims.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::LogicalReductionKeepDims);

ov::op::util::LogicalReductionKeepDims::LogicalReductionKeepDims(const ngraph::Output<ngraph::Node>& arg,
                                                                 const ngraph::Output<ngraph::Node>& reduction_axes,
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
    const PartialShape& axes_shape = get_input_partial_shape(1);
    const element::Type& axes_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this, data_et.compatible(element::boolean), "Element type of data input must be boolean.");

    NODE_VALIDATION_CHECK(this,
                          axes_et.is_integral_number(),
                          "Element type of axes input must be integer. Got: ",
                          axes_et);

    const Rank axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    PartialShape result_shape = infer_reduction_output_shape(m_keep_dims);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, data_et, result_shape);
}
