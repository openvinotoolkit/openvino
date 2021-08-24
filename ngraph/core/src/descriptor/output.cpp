// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/output.hpp"

#include <algorithm>

#include "ngraph/node.hpp"
#include "openvino/core/descriptor/input.hpp"

using namespace std;
using namespace ov;

descriptor::Output::Output(ngraph::Node* node, size_t index, const shared_ptr<Tensor>& tensor)
    : m_node(node),
      m_index(index),
      m_tensor(tensor) {}

// Add an input to the vector of inputs that use this output.
void descriptor::Output::add_input(Input* input) {
    // Keep the inputs in insertion order to keep sorts deterministic
    if (find(m_inputs.begin(), m_inputs.end(), input) == m_inputs.end()) {
        m_inputs.push_back(input);
    }
}

void descriptor::Output::remove_input(Input* input) {
    auto it = find(m_inputs.begin(), m_inputs.end(), input);
    if (it != m_inputs.end()) {
        m_inputs.erase(it);
    }
}

shared_ptr<ngraph::Node> descriptor::Output::get_node() const {
    return m_node->shared_from_this();
}

ngraph::Output<ngraph::Node> descriptor::Output::get_output() const {
    return get_node()->output(m_index);
}

descriptor::Tensor& descriptor::Output::get_tensor() const {
    return *m_tensor;
}

const ngraph::Shape& descriptor::Output::get_shape() const {
    return m_tensor->get_shape();
}

const PartialShape& descriptor::Output::get_partial_shape() const {
    return m_tensor->get_partial_shape();
}

const element::Type& descriptor::Output::get_element_type() const {
    return m_tensor->get_element_type();
}
