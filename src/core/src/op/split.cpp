// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include <numeric>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/split.hpp"
#include "split_shape_inference.hpp"

namespace ov {
namespace op {

namespace v1 {
namespace validate {
namespace {
bool axis_type(const element::Type& et) {
    return et.is_integral_number();
}
}  // namespace
}  // namespace validate

Split::Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits)
    : Op({data, axis}),
      m_num_splits{num_splits} {
    constructor_validate_and_infer_types();
}

bool Split::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Split_visit_attributes);
    visitor.on_attribute("num_splits", m_num_splits);
    return true;
}

void Split::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Split_validate_and_infer_types);
    const auto& axis_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          validate::axis_type(axis_et),
                          "Element type of 'axis' input must be integer. Got: ",
                          axis_et);

    NODE_VALIDATION_CHECK(this,
                          m_num_splits > 0,
                          "Attribute 'num_splits' must be greater than zero. Got: ",
                          m_num_splits);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    for (size_t i = 0; i < m_num_splits; ++i) {
        set_output_type(i, get_input_element_type(0), output_shapes[i]);
    }

    set_input_is_relevant_to_shape(0);
}

std::shared_ptr<Node> Split::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Split_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Split>(new_args.at(0), new_args.at(1), m_num_splits);
}

bool Split::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Split_evaluate);
    OPENVINO_ASSERT(outputs.size() == m_num_splits);

    const auto output_shapes =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs));
    const auto& axis_tensor = inputs[1];
    const auto result = validate::axis_type(axis_tensor.get_element_type());
    if (result) {
        const auto& data_tensor = inputs[0];

        auto outputs_data = std::vector<char*>(m_num_splits);
        {
            auto outputs_it = outputs.begin();
            auto outputs_data_it = outputs_data.begin();
            for (const auto& p_shape : output_shapes) {
                outputs_it->set_shape(p_shape.get_shape());
                *outputs_data_it = static_cast<char*>(outputs_it->data());
                ++outputs_it, ++outputs_data_it;
            }
        }

        auto axis = get_tensor_data_as<int64_t>(axis_tensor).front();
        axis = ov::util::normalize(axis, data_tensor.get_shape().size());

        ov::reference::split(static_cast<char*>(data_tensor.data()),
                             data_tensor.get_shape(),
                             data_tensor.get_element_type().size(),
                             axis,
                             m_num_splits,
                             outputs_data.data());
    }

    return result;
}

bool Split::has_evaluate() const {
    OV_OP_SCOPE(v1_Split_has_evaluate);
    return validate::axis_type(get_input_element_type(1));
}

bool Split::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Split::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Split::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    OPENVINO_ASSERT(output_symbols.size() == get_num_splits());

    return get_input_tensor(1).has_and_set_bound() && ov::util::default_symbol_evaluator(this, output_symbols);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
