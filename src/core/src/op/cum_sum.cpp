// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include "itt.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/constant.hpp"

using namespace std;

namespace ov {
op::v0::CumSum::CumSum(const Output<Node>& arg, const Output<Node>& axis, const bool exclusive, const bool reverse)
    : Op({arg, axis}),
      m_exclusive(exclusive),
      m_reverse(reverse) {
    constructor_validate_and_infer_types();
}

op::v0::CumSum::CumSum(const Output<Node>& arg, const bool exclusive, const bool reverse)
    : Op({arg, op::v0::Constant::create(element::i32, Shape{}, {0})}),
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

OPENVINO_SUPPRESS_DEPRECATED_START
shared_ptr<Node> op::v0::CumSum::get_default_value() const {
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
OPENVINO_SUPPRESS_DEPRECATED_END

namespace {
template <element::Type_t DATA_ET, element::Type_t AXIS_ET>
bool evaluate_cum_sum(TensorVector& outputs, const TensorVector& inputs, const bool exclusive, const bool reverse) {
    using data_t = fundamental_type_for<DATA_ET>;
    using axis_t = fundamental_type_for<AXIS_ET>;
    ngraph::runtime::reference::cumsum<data_t, axis_t>(inputs[0].data<data_t>(),
                                                       inputs[1].data<axis_t>(),
                                                       outputs[0].data<data_t>(),
                                                       inputs[0].get_shape(),
                                                       exclusive,
                                                       reverse);
    return true;
}

#define CUM_SUM_TYPE_CASE(a, ...)                                          \
    case element::Type_t::a: {                                             \
        OV_OP_SCOPE(OV_PP_CAT4(evaluate_cum_sum, _, a, AXIS_ET));          \
        return evaluate_cum_sum<element::Type_t::a, AXIS_ET>(__VA_ARGS__); \
    }

template <element::Type_t AXIS_ET>
bool evaluate(TensorVector& outputs, const TensorVector& inputs, const bool exclusive, const bool reverse) {
    switch (inputs[0].get_element_type()) {
        CUM_SUM_TYPE_CASE(f32, outputs, inputs, exclusive, reverse);
    default:
        return false;
    }
}

bool evaluate_cum_sum(TensorVector& outputs, const TensorVector& inputs, const bool exclusive, const bool reverse) {
    auto rc = true;
    switch (inputs[1].get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_cum_sum, i32, outputs, inputs, exclusive, reverse);
        NGRAPH_TYPE_CASE(evaluate_cum_sum, i64, outputs, inputs, exclusive, reverse);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v0::CumSum::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_CumSum_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2,
                    "Invalid size of inputs argument of evaluate method of CumSum operation. Provided: ",
                    inputs.size(),
                    ". Expected: 2");
    OPENVINO_ASSERT(outputs.size() == 1,
                    "Invalid size of outputs argument of evaluate method of CumSum operation. Provided: ",
                    outputs.size(),
                    ". Expected: 1");

    return evaluate_cum_sum(outputs, inputs, is_exclusive(), is_reverse());
}

bool op::v0::CumSum::has_evaluate() const {
    OV_OP_SCOPE(v0_CumSum_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    const auto& input_1_et = get_input_element_type(1);
    return input_0_et == element::f32 && (input_1_et == element::i32 || input_1_et == element::i64);
}
}  // namespace ov
