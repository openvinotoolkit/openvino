// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/input.hpp"

#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/tensor_util.hpp"
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
    Output* old_output = m_output;

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

    // Conditional bounds invalidation: only if bounds differ or new source doesn't have bounds
    // This ensures:
    // - OptimizeSymbolsUsedAsValues: Same bounds → no invalidation → optimization works
    // - AbsSinking: New Abs node has no bounds yet → invalidation → correct recalculation
    if (old_output != nullptr && m_node != nullptr) {
        const auto& old_tensor = old_output->get_tensor();
        const auto& new_tensor = new_output.get_tensor();

        const auto& old_lower = old_tensor.get_lower_value();
        const auto& old_upper = old_tensor.get_upper_value();
        const auto& new_lower = new_tensor.get_lower_value();
        const auto& new_upper = new_tensor.get_upper_value();

        bool old_has_bounds = old_lower && old_upper;
        bool new_has_bounds = new_lower && new_upper;

        // Invalidate if:
        // 1. Old had bounds but new doesn't have bounds (node was replaced with newly created one)
        // 2. Both have bounds but they differ
        bool should_invalidate = false;
        if (old_has_bounds && !new_has_bounds) {
            // New source doesn't have bounds yet (e.g., newly created Abs in AbsSinking)
            should_invalidate = true;
        } else if (old_has_bounds && new_has_bounds) {
            // Both have bounds - check if they differ
            bool bounds_differ = !ov::util::tensors_equal(old_lower, new_lower) ||
                                 !ov::util::tensors_equal(old_upper, new_upper);
            should_invalidate = bounds_differ;
        }

        if (should_invalidate) {
            for (size_t port = 0; port < m_node->get_output_size(); ++port) {
                ov::util::force_invalidate_bounds(m_node->get_output_tensor(port));
            }
        }
    }
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
