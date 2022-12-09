// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;

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
bool evaluate_cum_sum(const HostTensorPtr& output,
                      const HostTensorPtr& data,
                      const HostTensorPtr& axis,
                      const bool exclusive,
                      const bool reverse) {
    using data_t = fundamental_type_for<DATA_ET>;
    using axis_t = fundamental_type_for<AXIS_ET>;
    ngraph::runtime::reference::cumsum<data_t, axis_t>(data->get_data_ptr<data_t>(),
                                                       axis->get_data_ptr<axis_t>(),
                                                       output->get_data_ptr<data_t>(),
                                                       data->get_shape(),
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
bool evaluate(const HostTensorPtr& output,
              const HostTensorPtr& data,
              const HostTensorPtr& axis,
              const bool exclusive,
              const bool reverse) {
    switch (data->get_element_type()) {
        CUM_SUM_TYPE_CASE(bf16, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(f16, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(f32, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(f64, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(i4, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(i8, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(i16, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(i32, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(i64, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(u1, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(u4, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(u8, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(u16, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(u32, output, data, axis, exclusive, reverse);
        CUM_SUM_TYPE_CASE(u64, output, data, axis, exclusive, reverse);
    default:
        return false;
    }
}

bool evaluate_cum_sum(const HostTensorPtr& output,
                      const HostTensorPtr& data,
                      const HostTensorPtr& axis,
                      const bool exclusive,
                      const bool reverse) {
    auto rc = true;
    switch (axis->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_cum_sum, i32, output, data, axis, exclusive, reverse);
        NGRAPH_TYPE_CASE(evaluate_cum_sum, i64, output, data, axis, exclusive, reverse);
    default:
        return false;
    }
    return rc;
}
}  // namespace

bool op::v0::CumSum::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_CumSum_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 2), "Invalid CumSum input TensorVector.");
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1), "Invalid CumSum output TensorVector.");

    return evaluate_cum_sum(outputs[0], inputs[0], inputs[1], is_exclusive(), is_reverse());
}

bool op::v0::CumSum::has_evaluate() const {
    OV_OP_SCOPE(v0_CumSum_has_evaluate);
    const auto& input_1_element_type = get_input_element_type(1);
    return input_1_element_type == element::i32 || input_1_element_type == element::i64;
}
