// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/cum_sum.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::v3::CumSum);

ov::op::v3::CumSum::CumSum(const Output<Node>& arg, const Output<Node>& axis, const bool exclusive, const bool reverse)
    : Op({arg, axis}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

ov::op::v3::CumSum::CumSum(const Output<Node>& arg, const bool exclusive, const bool reverse)
    : Op({arg, op::v1::Constant::create(element::i32, ov::Shape{}, {0})}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

bool ov::op::v3::CumSum::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v3_CumSum_visit_attributes);
    visitor.on_attribute("exclusive", m_exclusive);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

void ov::op::v3::CumSum::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v3_CumSum_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));

    const auto& axis_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axis_type == element::i32 || axis_type == element::i64,
                          "axis element type must be either int64_t or int32_t but got (",
                          axis_type,
                          ").");

    // No axis input shape check for backward compatibility
}

shared_ptr<ov::Node> ov::op::v3::CumSum::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v3_CumSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2)
        return make_shared<op::v3::CumSum>(new_args.at(0), new_args.at(1), m_exclusive, m_reverse);
    else {
        return make_shared<op::v3::CumSum>(new_args.at(0), m_exclusive, m_reverse);
    }
}

NGRAPH_SUPPRESS_DEPRECATED_START
shared_ptr<ov::Node> ov::op::v3::CumSum::get_default_value() const {
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
NGRAPH_SUPPRESS_DEPRECATED_END
