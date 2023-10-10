// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/clamp.hpp"

#include <ngraph/validation_util.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/util.hpp"
#include "openvino/reference/clamp.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace clamp {
namespace {
template <element::Type_t ET, typename T>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, T min, T max, size_t count) {
    ov::reference::clamp<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), min, max, count);
    return true;
}

bool evaluate_clamp(const HostTensorPtr& arg, const HostTensorPtr& out, double min, double max) {
    size_t count = shape_size(arg->get_shape());
    auto ceil_func = [](double x) {
        return ceil(x);
    };
    auto floor_func = [](double x) {
        return floor(x);
    };

    bool rc = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    switch (arg->get_element_type()) {
        TYPE_CASE(i8)
        (arg, out, double_to_int<int8_t>(min, ceil_func), double_to_int<int8_t>(max, floor_func), count);
        break;
        TYPE_CASE(i16)
        (arg, out, double_to_int<int16_t>(min, ceil_func), double_to_int<int16_t>(max, floor_func), count);
        break;
        TYPE_CASE(i32)
        (arg, out, double_to_int<int32_t>(min, ceil_func), double_to_int<int32_t>(max, floor_func), count);
        break;
        TYPE_CASE(i64)
        (arg, out, double_to_int<int64_t>(min, ceil_func), double_to_int<int64_t>(max, floor_func), count);
        break;
        TYPE_CASE(u8)
        (arg, out, double_to_int<uint8_t>(min, ceil_func), double_to_int<uint8_t>(max, floor_func), count);
        break;
        TYPE_CASE(u16)
        (arg, out, double_to_int<uint16_t>(min, ceil_func), double_to_int<uint16_t>(max, floor_func), count);
        break;
        TYPE_CASE(u32)
        (arg, out, double_to_int<uint32_t>(min, ceil_func), double_to_int<uint32_t>(max, floor_func), count);
        break;
        TYPE_CASE(u64)
        (arg, out, double_to_int<uint64_t>(min, ceil_func), double_to_int<uint64_t>(max, floor_func), count);
        break;
        TYPE_CASE(f16)(arg, out, static_cast<float16>(min), static_cast<float16>(max), count);
        break;
        TYPE_CASE(bf16)
        (arg, out, static_cast<bfloat16>(min), static_cast<bfloat16>(max), count);
        break;
        TYPE_CASE(f32)(arg, out, static_cast<float>(min), static_cast<float>(max), count);
        break;
    default:
        rc = false;
        break;
    }
    return rc;
    OPENVINO_SUPPRESS_DEPRECATED_END
}
}  // namespace
}  // namespace clamp

bool op::v0::Clamp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Clamp_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return clamp::evaluate_clamp(inputs[0], outputs[0], get_min(), get_max());
}

bool op::v0::Clamp::has_evaluate() const {
    OV_OP_SCOPE(v0_Clamp_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::bf16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

op::Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : util::UnaryElementwiseArithmetic(data),
      m_min{min},
      m_max{max} {
    constructor_validate_and_infer_types();
}

void op::Clamp::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Clamp_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et.is_integral_number() || input_et.is_real(),
                          "Input element type must be numeric. Got: ",
                          input_et);
    NODE_VALIDATION_CHECK(this,
                          m_min <= m_max,
                          "Attribute 'min' must be less or equal than 'max'. Got: ",
                          m_min,
                          " and ",
                          m_max);
    set_output_type(0, input_et, get_input_partial_shape(0));
}

shared_ptr<Node> op::Clamp::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Clamp_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the Clamp op but got ",
                          new_args.size());

    return make_shared<Clamp>(new_args.at(0), m_min, m_max);
}

bool op::Clamp::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Clamp_visit_attributes);
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    return true;
}

bool op::Clamp::evaluate_lower(ov::TensorVector& output_values) const {
    return ov::default_lower_bound_evaluator(this, output_values);
}

bool op::Clamp::evaluate_upper(ov::TensorVector& output_values) const {
    return ov::default_upper_bound_evaluator(this, output_values);
}
