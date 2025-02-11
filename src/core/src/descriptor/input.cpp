// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/input.hpp"

#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shared_node_info.hpp"

ov::descriptor::Input::Input(ov::Node* node, size_t index, Output& output)
    : m_node(node),
      m_index(index),
      m_output(&output),
      m_is_relevant_to_shape(false),
      m_is_relevant_to_value(true) {
    m_src_node = std::shared_ptr<ov::Node>(output.get_node());
    output.add_input(this);
}

ov::descriptor::Input::Input(ov::Node* node, size_t index)
    : m_node(node),
      m_index(index),
      m_output(nullptr),
      m_is_relevant_to_shape(false),
      m_is_relevant_to_value(true) {}

ov::descriptor::Input::~Input() {
    remove_output();
}

void ov::descriptor::Input::replace_output(Output& new_output) {
    if (m_output != nullptr) {
        m_output->remove_input(this);
    }
    new_output.add_input(this);
    m_output = &new_output;
    m_src_node = std::shared_ptr<ov::Node>(new_output.get_node());

    // Output replacement may change the topological order of nodes,
    // so we have to reset cache by setting a flag into shared node info.
    for_each(m_node->m_shared_rt_info.cbegin(),
             m_node->m_shared_rt_info.cend(),
             [](const std::shared_ptr<SharedRTInfo>& info) {
                 info->set_use_topological_cache(false);
             });
}

void ov::descriptor::Input::replace_output(const std::shared_ptr<ov::Node>& node, size_t i) {
    replace_output(node->m_outputs.at(i));
}

void ov::descriptor::Input::remove_output() {
    if (m_output != nullptr) {
        m_output->remove_input(this);
        m_src_node = nullptr;
        m_output = nullptr;
    }
}

std::shared_ptr<ov::Node> ov::descriptor::Input::get_node() const {
    return m_node->shared_from_this();
}

const ov::descriptor::Tensor& ov::descriptor::Input::get_tensor() const {
    return m_output->get_tensor();
}

ov::descriptor::Tensor& ov::descriptor::Input::get_tensor() {
    return m_output->get_tensor();
}

const ov::Shape& ov::descriptor::Input::get_shape() const {
    return m_output->get_shape();
}

const ov::PartialShape& ov::descriptor::Input::get_partial_shape() const {
    return m_output->get_partial_shape();
}

const ov::element::Type& ov::descriptor::Input::get_element_type() const {
    return m_output->get_element_type();
}
