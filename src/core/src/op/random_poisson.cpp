// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_poisson.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/random_poisson.hpp"

namespace ov {
namespace op {
namespace random_poisson {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;
    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const v17::RandomPoisson* node,
                             const Tensor& input,
                             Tensor& output,
                             const Shape& input_shape,
                             const Shape& output_shape,
                             uint64_t global_seed,
                             uint64_t op_seed,
                             std::pair<uint64_t, uint64_t> prev_state,
                             ov::op::PhiloxAlignment alignment) {
        auto result = reference::random_poisson<T>(input.data<const T>(),
                                                   output.data<T>(),
                                                   input_shape,
                                                   output_shape,
                                                   global_seed,
                                                   op_seed,
                                                   prev_state,
                                                   alignment);
        node->set_state(result);
        return true;
    }
};
}  // namespace random_poisson

namespace validate {
inline bool input_et(const element::Type& et) {
    return et == element::bf16 || et == element::f16 || et == element::f32 || et == element::f64;
}
inline bool alignment(const PhiloxAlignment& alignment) {
    return alignment == PhiloxAlignment::TENSORFLOW || alignment == PhiloxAlignment::PYTORCH;
}
}  // namespace validate

namespace v17 {

RandomPoisson::RandomPoisson(const Output<Node>& input,
                             uint64_t global_seed,
                             uint64_t op_seed,
                             PhiloxAlignment alignment)
    : Op({input}),
      m_global_seed(global_seed),
      m_op_seed(op_seed),
      m_alignment(alignment) {
    constructor_validate_and_infer_types();
}

void RandomPoisson::validate_and_infer_types() {
    // Validate the input is of type float and not an integer
    // output is same type as the input, and the shape is the same as the input

    OV_OP_SCOPE(v17_RandomPoisson_validate_and_infer_types);
    const auto& input_shape = get_input_partial_shape(0);
    const auto& input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || validate::input_et(input_element_type),
                          "Input tensor must be of type float, bf16, f16, f32, or f64. or dynamic.");
    NODE_VALIDATION_CHECK(this,
                          input_shape.rank().is_dynamic() || input_shape.rank().get_length() > 0,
                          "RandomPoisson: scalars (rank 0) are not supported.");
    NODE_VALIDATION_CHECK(this,
                          validate::alignment(get_alignment()),
                          "Unknown alignment mode provided to RandomPoisson.");
    // set graph metadata for output port
    set_output_type(0, input_element_type, input_shape);
}

bool RandomPoisson::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_RandomPoisson_visit_attributes);
    visitor.on_attribute("global_seed", m_global_seed);
    visitor.on_attribute("op_seed", m_op_seed);
    visitor.on_attribute("alignment", m_alignment);
    return true;
}

std::shared_ptr<Node> RandomPoisson::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_RandomPoisson_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto ru_copy = std::make_shared<RandomPoisson>(new_args.at(0), m_global_seed, m_op_seed, m_alignment);
    ru_copy->m_state = this->m_state;
    return ru_copy;
}

/// \return Turns off constant folding for RandomPoisson operation.
bool RandomPoisson::can_constant_fold(const OutputVector& input_values) const {
    return false;
}

/// \return The global seed value.
uint64_t RandomPoisson::get_global_seed() const {
    return m_global_seed;
}

void RandomPoisson::set_global_seed(uint64_t seed) {
    m_global_seed = seed;
}

/// \return The operational seed value.
uint64_t RandomPoisson::get_op_seed() const {
    return m_op_seed;
}

void RandomPoisson::set_op_seed(uint64_t seed2) {
    m_op_seed = seed2;
}

/// \return The state value.
std::pair<uint64_t, uint64_t> RandomPoisson::get_state() const {
    return m_state;
}

/// \brief Set the state value.
void RandomPoisson::set_state(std::pair<uint64_t, uint64_t> state) const {
    m_state = state;
}

/// \return The alignment mode.
PhiloxAlignment RandomPoisson::get_alignment() const {
    return m_alignment;
}

void RandomPoisson::set_alignment(ov::op::PhiloxAlignment alignment) {
    m_alignment = alignment;
}

bool RandomPoisson::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_RandomPoisson_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(inputs[0].get_element_type().is_real(), "Input tensor must be of type float, f16, f32, or f64.");
    outputs[0].set_shape(inputs[0].get_shape());
    OPENVINO_ASSERT(inputs[0].get_shape() == outputs[0].get_shape());
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v17_RandomPoisson_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f16, f64),
                                      random_poisson::Evaluate,
                                      inputs[0].get_element_type(),
                                      this,
                                      inputs[0],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      outputs[0].get_shape(),
                                      get_global_seed(),
                                      get_op_seed(),
                                      get_state(),
                                      get_alignment());
}

bool RandomPoisson::has_evaluate() const {
    OV_OP_SCOPE(v17_RandomPoisson_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}
}  // namespace v17
}  // namespace op
}  // namespace ov