// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/clamp.hpp"

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/clamp.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace op {
namespace clamp {

// Make it part of reference/convert.hpp (requires to move compare.hpp to reference from shape inference)
template <class TO, class FROM>
TO convert(const FROM value) {
    if (cmp::lt(value, std::numeric_limits<TO>::min())) {
        return std::numeric_limits<TO>::lowest();
    } else if (cmp::gt(value, std::numeric_limits<TO>::max())) {
        return std::numeric_limits<TO>::max();
    } else {
        return static_cast<TO>(value);
    }
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T min_as(const double value) {
    return convert<T, double>(std::ceil(value));
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T min_as(const double value) {
    return static_cast<T>(value);
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T max_as(const double value) {
    return convert<T, double>(std::floor(value));
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T max_as(const double value) {
    return static_cast<T>(value);
}

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg, Tensor& out, const double min, const double max, const size_t count) {
        reference::clamp(arg.data<const T>(), out.data<T>(), min_as<T>(min), max_as<T>(max), count);
        return true;
    }
};
}  // namespace clamp

namespace v0 {
bool Clamp::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Clamp_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& in_shape = inputs[0].get_shape();
    outputs[0].set_shape(in_shape);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Clamp_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i16, i32, i64, u8, u16, u32, u64),
                                      clamp::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      get_min(),
                                      get_max(),
                                      shape_size(in_shape));
}

bool Clamp::has_evaluate() const {
    OV_OP_SCOPE(v0_Clamp_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : util::UnaryElementwiseArithmetic(data),
      m_min{min},
      m_max{max} {
    constructor_validate_and_infer_types();
}

void Clamp::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Clamp_validate_and_infer_types);
    const auto& input_et = get_input_element_type(0);
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

std::shared_ptr<Node> Clamp::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Clamp_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Clamp>(new_args.at(0), get_min(), get_max());
}

bool Clamp::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Clamp_visit_attributes);
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    return true;
}

bool Clamp::evaluate_lower(ov::TensorVector& output_values) const {
    return ov::default_lower_bound_evaluator(this, output_values);
}

bool Clamp::evaluate_upper(ov::TensorVector& output_values) const {
    return ov::default_upper_bound_evaluator(this, output_values);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
