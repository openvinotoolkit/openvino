// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_input.hpp"

#include "openvino/core/node.hpp"

namespace ov {
Input<Node>::Input(Node* node, size_t index) : m_node(node), m_index(index) {
    OPENVINO_ASSERT(m_node, "Cannot create ov::Input<ov::Node> from nullptr!");
}

Node* Input<Node>::get_node() const {
    return m_node;
}
size_t Input<Node>::get_index() const {
    return m_index;
}
const element::Type& Input<Node>::get_element_type() const {
    return m_node->get_input_element_type(m_index);
}

const Shape& Input<Node>::get_shape() const {
    return m_node->get_input_shape(m_index);
}
const PartialShape& Input<Node>::get_partial_shape() const {
    return m_node->get_input_partial_shape(m_index);
}

Output<Node> Input<Node>::get_source_output() const {
    auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
    return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
}

descriptor::Tensor& Input<Node>::get_tensor() const {
    return m_node->m_inputs.at(m_index).get_output().get_tensor();
}

std::shared_ptr<descriptor::Tensor> Input<Node>::get_tensor_ptr() const {
    return m_node->m_inputs.at(m_index).get_output().get_tensor_ptr();
}

bool Input<Node>::get_is_relevant_to_shapes() const {
    return m_node->m_inputs.at(m_index).get_is_relevant_to_shape();
}

bool Input<Node>::get_is_relevant_to_values() const {
    return m_node->m_inputs.at(m_index).get_is_relevant_to_value();
}

void Input<Node>::replace_source_output(const Output<Node>& new_source_output) const {
    m_node->m_inputs.at(m_index).replace_output(new_source_output.get_node_shared_ptr(), new_source_output.get_index());
}

bool Input<Node>::operator==(const Input& other) const {
    return m_node == other.m_node && m_index == other.m_index;
}

bool Input<Node>::operator!=(const Input& other) const {
    return !(*this == other);
}

bool Input<Node>::operator<(const Input& other) const {
    return m_node->get_instance_id() < other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index < other.m_index);
}

bool Input<Node>::operator>(const Input& other) const {
    return m_node->get_instance_id() > other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index > other.m_index);
}

bool Input<Node>::operator<=(const Input& other) const {
    return !(*this > other);
}
bool Input<Node>::operator>=(const Input& other) const {
    return !(*this < other);
}
Input<const Node>::Input(const Node* node, size_t index) : m_node(node), m_index(index) {
    OPENVINO_ASSERT(m_node, "Cannot create ov::Input<const ov::Node> from nullptr!");
}

RTMap& Input<Node>::get_rt_info() {
    return m_node->m_inputs.at(m_index).get_rt_info();
}

const RTMap& Input<Node>::get_rt_info() const {
    return m_node->m_inputs.at(m_index).get_rt_info();
}

const RTMap& Input<const Node>::get_rt_info() const {
    return m_node->m_inputs.at(m_index).get_rt_info();
}

const Node* Input<const Node>::get_node() const {
    return m_node;
}
size_t Input<const Node>::get_index() const {
    return m_index;
}
const element::Type& Input<const Node>::get_element_type() const {
    return m_node->get_input_element_type(m_index);
}
const Shape& Input<const Node>::get_shape() const {
    return m_node->get_input_shape(m_index);
}
const PartialShape& Input<const Node>::get_partial_shape() const {
    return m_node->get_input_partial_shape(m_index);
}

Output<Node> Input<const Node>::get_source_output() const {
    auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
    return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
}

descriptor::Tensor& Input<const Node>::get_tensor() const {
    return m_node->m_inputs.at(m_index).get_output().get_tensor();
}

std::shared_ptr<descriptor::Tensor> Input<const Node>::get_tensor_ptr() const {
    return m_node->m_inputs.at(m_index).get_output().get_tensor_ptr();
}

bool Input<const Node>::get_is_relevant_to_shapes() const {
    return m_node->m_inputs.at(m_index).get_is_relevant_to_shape();
}

bool Input<const Node>::get_is_relevant_to_values() const {
    return m_node->m_inputs.at(m_index).get_is_relevant_to_value();
}

bool Input<const Node>::operator==(const Input& other) const {
    return m_node == other.m_node && m_index == other.m_index;
}

bool Input<const Node>::operator!=(const Input& other) const {
    return !(*this == other);
}
bool Input<const Node>::operator<(const Input& other) const {
    return m_node->get_instance_id() < other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index < other.m_index);
}

bool Input<const Node>::operator>(const Input& other) const {
    return m_node->get_instance_id() > other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index > other.m_index);
}

bool Input<const Node>::operator<=(const Input& other) const {
    return !(*this > other);
}
bool Input<const Node>::operator>=(const Input& other) const {
    return !(*this < other);
}
std::ostream& operator<<(std::ostream& out, const Input<Node>& input) {
    return input.get_node()->write_description(out, 0)
           << ".input(" << input.get_index() << "):" << input.get_element_type() << input.get_partial_shape();
}

std::ostream& operator<<(std::ostream& out, const Input<const Node>& input) {
    return input.get_node()->write_description(out, 0)
           << ".input(" << input.get_index() << "):" << input.get_element_type() << input.get_partial_shape();
}
}  // namespace ov
