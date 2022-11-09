// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/cum_sum.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::CumSum);

op::v0::CumSum::CumSum(const Output<Node>& arg, const Output<Node>& axis, const bool exclusive, const bool reverse)
    : Op({arg, axis}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

op::v0::CumSum::CumSum(const Output<Node>& arg, const bool exclusive, const bool reverse)
    : Op({arg, op::v0::Constant::create(element::i32, ov::Shape{}, {0})}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

bool op::v0::CumSum::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_CumSum_visit_attributes);
    visitor.on_attribute("exclusive", m_exclusive);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

void op::v0::CumSum::validate_and_infer_types() {
    OV_OP_SCOPE(v0_CumSum_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));

    const auto& axis_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axis_type == element::i32 || axis_type == element::i64,
                          "axis element type must be either int64_t or int32_t but got (",
                          axis_type,
                          ").");

    // No axis input shape check for backward compatibility
}

shared_ptr<Node> op::v0::CumSum::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_CumSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2)
        return make_shared<op::v0::CumSum>(new_args.at(0), new_args.at(1), m_exclusive, m_reverse);
    else {
        return make_shared<op::v0::CumSum>(new_args.at(0), m_exclusive, m_reverse);
    }
}

NGRAPH_SUPPRESS_DEPRECATED_START
shared_ptr<Node> op::v0::CumSum::get_default_value() const {
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
NGRAPH_SUPPRESS_DEPRECATED_END
