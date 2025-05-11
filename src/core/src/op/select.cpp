// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include <memory>

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/reference/select.hpp"
#include "select_shape_inference.hpp"

namespace ov {
namespace op {
namespace select {
struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t DATA_ET,
              class DT = fundamental_type_for<DATA_ET>,
              class BT = fundamental_type_for<element::Type_t::boolean>>
    static result_type visit(const Tensor& cond_input,
                             const Tensor& then_input,
                             const Tensor& else_input,
                             Tensor& output,
                             const Shape& cond_shape,
                             const Shape& then_shape,
                             const Shape& else_shape,
                             const AutoBroadcastSpec& auto_broadcast) {
        using namespace ov::element;
        reference::select(cond_input.data<const BT>(),
                          then_input.data<const DT>(),
                          else_input.data<const DT>(),
                          output.data<DT>(),
                          cond_shape,
                          then_shape,
                          else_shape,
                          auto_broadcast);
        return true;
    }
};
}  // namespace select

namespace v1 {
Select::Select(const Output<Node>& arg0,
               const Output<Node>& arg1,
               const Output<Node>& arg2,
               const AutoBroadcastSpec& auto_broadcast)
    : Op({arg0, arg1, arg2}),
      m_auto_broadcast(auto_broadcast) {
    constructor_validate_and_infer_types();
}

void Select::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Select_validate_and_infer_types);
    // Condition element type check
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() || get_input_element_type(0) == element::boolean,
                          "Argument 0 must have boolean element type (element type: ",
                          get_input_element_type(0),
                          ").");

    // Then/Else element type check
    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(1), get_input_element_type(2)),
                          "Argument 1 and 2 element types must match.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> Select::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Select_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::Select>(new_args.at(0), new_args.at(1), new_args.at(2), m_auto_broadcast);
}

bool Select::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Select_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

bool Select::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Select_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto output_shape = shape_infer(this, ov::util::get_tensors_partial_shapes(inputs)).front().to_shape();
    auto& output = outputs[0];
    output.set_shape(output_shape);

    const auto& cond_input = inputs[0];
    const auto& then_input = inputs[1];
    const auto& else_input = inputs[2];

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Select_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64),
                                      select::Evaluate,
                                      then_input.get_element_type(),
                                      cond_input,
                                      then_input,
                                      else_input,
                                      output,
                                      cond_input.get_shape(),
                                      then_input.get_shape(),
                                      else_input.get_shape(),
                                      m_auto_broadcast);
}

bool Select::evaluate_lower(TensorVector& output_values) const {
    return get_input_tensor(0).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Select::evaluate_upper(TensorVector& output_values) const {
    return get_input_tensor(0).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Select::has_evaluate() const {
    OV_OP_SCOPE(v1_Select_has_evaluate);
    switch (get_output_element_type(0)) {
    case element::boolean:
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
}  // namespace v1
}  // namespace op
}  // namespace ov
