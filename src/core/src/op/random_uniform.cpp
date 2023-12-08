// Copyright (C) 2018-2023 Intel Corporation
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
}  // namespace validate

RandomUniform::RandomUniform(const Output<Node>& out_shape,
                             const Output<Node>& min_val,
                             const Output<Node>& max_val,
                             const ov::element::Type& out_type,
                             uint64_t global_seed,
                             uint64_t op_seed)
    : Op({out_shape, min_val, max_val}),
      m_output_type(out_type),
      m_global_seed(global_seed),
      m_op_seed(op_seed) {
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
    NODE_VALIDATION_CHECK(this, min_et == max_et, "'min_val' should have the same type as 'max_val'.");
    NODE_VALIDATION_CHECK(this,
                          validate::out_et(out_et) && (out_et == min_et),
                          "'min_val' and 'max_val' should have the same type as 'out_type' attribute.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, out_et, output_shapes.front());
}

bool RandomUniform::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_RandomUniform_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("op_seed", m_op_seed);
    visitor.on_attribute("global_seed", m_global_seed);
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
                                                       m_op_seed);
    ru_copy->m_state = this->m_state;
    return ru_copy;
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

    auto state = ov::reference::random_uniform(out_dims.data(),
                                               static_cast<const char*>(inputs[1].data()),
                                               static_cast<const char*>(inputs[2].data()),
                                               static_cast<char*>(outputs[0].data()),
                                               inputs[0].get_shape(),
                                               get_out_type(),
                                               get_global_seed(),
                                               get_op_seed(),
                                               m_state);

    // Update RandomUniform state
    std::lock_guard<std::mutex> guard(m_state_mutex);
    m_state = state;
    return true;
}

bool RandomUniform::has_evaluate() const {
    OV_OP_SCOPE(v8_RandomUniform_has_evaluate);
    return validate::shape_et(get_input_element_type(0)) && validate::out_et(get_out_type());
}
}  // namespace v8
}  // namespace op
}  // namespace ov
