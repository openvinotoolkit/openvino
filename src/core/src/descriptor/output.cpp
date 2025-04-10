// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/output.hpp"

#include <algorithm>

#include "openvino/core/descriptor/input.hpp"
#include "openvino/core/node.hpp"

ov::descriptor::Output::Output(ov::Node* node, size_t index, const std::shared_ptr<Tensor>& tensor)
    : m_node(node),
      m_index(index),
      m_tensor(tensor) {}

// Add an input to the vector of inputs that use this output.
void ov::descriptor::Output::add_input(Input* input) {
    // Keep the inputs in insertion order to keep sorts deterministic
    if (find(m_inputs.begin(), m_inputs.end(), input) == m_inputs.end()) {
        m_inputs.push_back(input);
    }
}

void ov::descriptor::Output::remove_input(Input* input) {
    auto it = find(m_inputs.begin(), m_inputs.end(), input);
    if (it != m_inputs.end()) {
        m_inputs.erase(it);
    }
}

std::shared_ptr<ov::Node> ov::descriptor::Output::get_node() const {
    return m_node->shared_from_this();
}

ov::Output<ov::Node> ov::descriptor::Output::get_output() const {
    return get_node()->output(m_index);
}

ov::descriptor::Tensor& ov::descriptor::Output::get_tensor() const {
    return *m_tensor;
}

const ov::Shape& ov::descriptor::Output::get_shape() const {
    return m_tensor->get_shape();
}

const ov::PartialShape& ov::descriptor::Output::get_partial_shape() const {
    return m_tensor->get_partial_shape();
}

const ov::element::Type& ov::descriptor::Output::get_element_type() const {
    return m_tensor->get_element_type();
}
