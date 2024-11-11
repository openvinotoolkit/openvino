// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/random_uniform.hpp"

#include "itt.hpp"
#include "openvino/op/random_uniform.hpp"
#include "random_uniform_shape_inference.hpp"

namespace ov {
namespace op {
namespace v8 {
namespace validate {
inline bool shape_et(const element::Type& et) {
    return (et == element::i32) || (et == element::i64);
}

inline bool out_et(const element::Type& et) {
    return et.is_real() || shape_et(et);
}

inline bool alignment(const PhiloxAlignment& alignment) {
    return alignment == PhiloxAlignment::TENSORFLOW || alignment == PhiloxAlignment::PYTORCH;
}
}  // namespace validate

RandomUniform::RandomUniform(const Output<Node>& out_shape,
                             const Output<Node>& min_val,
                             const Output<Node>& max_val,
                             const ov::element::Type& out_type,
                             uint64_t global_seed,
                             uint64_t op_seed,
                             PhiloxAlignment alignment)
    : Op({out_shape, min_val, max_val}),
      m_output_type(out_type),
      m_global_seed(global_seed),
      m_op_seed(op_seed),
      m_alignment(alignment) {
    constructor_validate_and_infer_types();
}

void RandomUniform::validate_and_infer_types() {
    OV_OP_SCOPE(v8_RandomUniform_validate_and_infer_types);

    const auto& shape_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          shape_et.is_dynamic() || validate::shape_et(shape_et),
                          "Type of the input should be int32 or int64.");
    const auto& min_et = get_input_element_type(1);
    const auto& max_et = get_input_element_type(2);
    const auto& out_et = get_out_type();
    const auto& alignment = get_alignment();

    NODE_VALIDATION_CHECK(this, min_et == max_et, "'min_val' should have the same type as 'max_val'.");
    NODE_VALIDATION_CHECK(this,
                          validate::out_et(out_et) && (out_et == min_et),
                          "'min_val' and 'max_val' should have the same type as 'out_type' attribute.");
    NODE_VALIDATION_CHECK(this, validate::alignment(alignment), "Unknown alignment mode provided to RandomUniform.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, out_et, output_shapes.front());
}

bool RandomUniform::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_RandomUniform_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("op_seed", m_op_seed);
    visitor.on_attribute("global_seed", m_global_seed);
    visitor.on_attribute("alignment", m_alignment);
    return true;
}

std::shared_ptr<Node> RandomUniform::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_RandomUniform_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto ru_copy = std::make_shared<v8::RandomUniform>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       m_output_type,
                                                       m_global_seed,
                                                       m_op_seed,
                                                       m_alignment);
    ru_copy->m_state = this->m_state;
    return ru_copy;
}

/// \return Turns off constant folding for RandomUniform operation.
bool RandomUniform::can_constant_fold(const OutputVector& input_values) const {
    return false;
}

/// \return The output tensor type.
const ov::element::Type& RandomUniform::get_out_type() const {
    return m_output_type;
}

void RandomUniform::set_out_type(const ov::element::Type& output_type) {
    m_output_type = output_type;
}

/// \return The global seed value.
uint64_t RandomUniform::get_global_seed() const {
    return m_global_seed;
}
void RandomUniform::set_global_seed(uint64_t seed) {
    m_global_seed = seed;
}

/// \return The operational seed value.
uint64_t RandomUniform::get_op_seed() const {
    return m_op_seed;
}
void RandomUniform::set_op_seed(uint64_t seed2) {
    m_op_seed = seed2;
}

/// \return The state value.
std::pair<uint64_t, uint64_t> RandomUniform::get_state() const {
    return m_state;
}

/// \return The alignment mode.
PhiloxAlignment RandomUniform::get_alignment() const {
    return m_alignment;
}

bool RandomUniform::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v8_RandomUniform_evaluate);

    auto input_shapes = std::vector<PartialShape>();
    input_shapes.reserve(inputs.size());
    for (auto& t : inputs) {
        input_shapes.emplace_back(t.get_shape());
    }
    const auto out_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();
    const auto out_dims = std::vector<uint64_t>(out_shape.begin(), out_shape.end());

    const auto& t_out = get_out_type();
    OPENVINO_ASSERT(validate::out_et(t_out), "Unsupported type of RandomUniform: " + t_out.get_type_name());

    outputs[0].set_shape(out_shape);

    m_state = ov::reference::random_uniform(out_dims.data(),
                                            static_cast<const char*>(inputs[1].data()),
                                            static_cast<const char*>(inputs[2].data()),
                                            static_cast<char*>(outputs[0].data()),
                                            inputs[0].get_shape(),
                                            get_out_type(),
                                            get_global_seed(),
                                            get_op_seed(),
                                            m_state,
                                            m_alignment);
    return true;
}

bool RandomUniform::has_evaluate() const {
    OV_OP_SCOPE(v8_RandomUniform_has_evaluate);
    return validate::shape_et(get_input_element_type(0)) && validate::out_et(get_out_type());
}
}  // namespace v8
}  // namespace op
}  // namespace ov
