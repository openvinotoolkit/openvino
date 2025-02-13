// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/range.hpp"
#include "range_shape_inference.hpp"

namespace ov {
namespace op {
namespace range {

#define RANGE_ET_LIST f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const double start, const double step, const size_t count, Tensor& out) {
        reference::range(static_cast<T>(start), static_cast<T>(step), count, out.data<T>());
        return true;
    }

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& start, const Tensor& step, const size_t count, Tensor& out) {
        reference::range(*start.data<const T>(), *step.data<const T>(), count, out.data<T>());
        return true;
    }
};

namespace {
bool is_input_valid_et(const element::Type& et) {
    switch (et) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
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
}  // namespace
}  // namespace range

namespace v4 {
Range::Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step, element::Type output_type)
    : Op({start, stop, step}),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool Range::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_Range_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void Range::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Range_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type.is_integral_number() || m_output_type.is_real(),
                          "output tensor type should be a numeric type. Got: ",
                          m_output_type);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_integral_number() || get_input_element_type(0).is_real(),
                          "'start' input scalar should be a numeric type. Got: ",
                          get_input_element_type(0));
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number() || get_input_element_type(1).is_real(),
                          "'stop' input scalar should be a numeric type. Got: ",
                          get_input_element_type(1));
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number() || get_input_element_type(2).is_real(),
                          "'step' input scalar should be a numeric type. Got: ",
                          get_input_element_type(2));

    std::vector<PartialShape> input_shapes;
    for (size_t i = 0; i < get_input_size(); i++)
        input_shapes.push_back(get_input_partial_shape(i));

    const auto result_shapes = shape_infer(this, input_shapes);

    set_output_type(0, m_output_type, result_shapes[0]);
}

std::shared_ptr<Node> Range::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Range_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Range>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type);
}

bool Range::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_Range_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto out_shape =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs))[0].to_shape();
    auto& out = outputs[0];
    out.set_shape(out_shape);

    const auto start = get_tensor_data_as<double>(inputs[0])[0];
    const auto step = get_tensor_data_as<double>(inputs[2])[0];

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v4_Range_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      RANGE_ET_LIST,
                                      range::Evaluate,
                                      out.get_element_type(),
                                      start,
                                      step,
                                      shape_size(out_shape),
                                      out);
}

bool Range::has_evaluate() const {
    OV_OP_SCOPE(v4_Range_has_evaluate);
    return range::is_input_valid_et(get_input_element_type(0));
}
}  // namespace v4

namespace v0 {

Range::Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step) : Op({start, stop, step}) {
    constructor_validate_and_infer_types();
}

bool Range::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Range_visit_attributes);
    return true;
}

void Range::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Range_validate_and_infer_types);
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    auto result_et = element::dynamic;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)),
                          "Element types for start, stop, and step do not match.");

    NODE_VALIDATION_CHECK(this,
                          result_et != element::boolean,
                          "Element type for start, stop, and step, must not be boolean.");

    NODE_VALIDATION_CHECK(
        this,
        result_et != element::Type_t::u1 && result_et != element::Type_t::i4 && result_et != element::Type_t::u4,
        "Internal OpenVINO error: unsupported element type: ",
        result_et);

    if (result_et == element::Type_t::dynamic) {
        set_output_type(0, result_et, ov::PartialShape::dynamic(1));
    } else {
        std::vector<PartialShape> input_shapes;
        for (size_t i = 0; i < get_input_size(); i++)
            input_shapes.push_back(get_input_partial_shape(i));

        const auto result_shapes = shape_infer(this, input_shapes);

        set_output_type(0, result_et, result_shapes[0]);
    }
}

std::shared_ptr<Node> Range::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Range_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Range>(new_args.at(0), new_args.at(1), new_args.at(2));
}

bool Range::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Range_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto out_shape =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs))[0].to_shape();
    const auto& start = inputs[0];
    const auto& step = inputs[2];

    auto& out = outputs[0];
    out.set_shape(out_shape);

    using namespace ov::element;
    return IF_TYPE_OF(v0_Range_evaluate,
                      RANGE_ET_LIST,
                      range::Evaluate,
                      out.get_element_type(),
                      start,
                      step,
                      shape_size(out_shape),
                      out);
}

bool Range::has_evaluate() const {
    OV_OP_SCOPE(v0_Range_has_evaluate);
    return range::is_input_valid_et(get_input_element_type(0));
}
}  // namespace v0
}  // namespace op
}  // namespace ov
