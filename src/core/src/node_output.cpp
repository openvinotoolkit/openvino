// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_output.hpp"

#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
Output<Node>::Output(Node* node, size_t index) : m_index(index) {
    OPENVINO_ASSERT(node, "Cannot create ov::Output<ov::Node> from nullptr!");
    m_node = node->shared_from_this();
}

Output<Node>::Output(const std::shared_ptr<Node>& node, size_t index) : m_node(node), m_index(index) {}

void Output<Node>::reset() {
    m_node.reset();
    m_index = 0;
}

Output<Node> Output<Node>::for_node(const std::shared_ptr<Node>& node) {
    return Output(node, m_index);
}
Node* Output<Node>::get_node() const {
    return m_node.get();
}
std::shared_ptr<Node> Output<Node>::get_node_shared_ptr() const {
    return m_node;
}
size_t Output<Node>::get_index() const {
    return m_index;
}
descriptor::Tensor& Output<Node>::get_tensor() const {
    return m_node->m_outputs.at(m_index).get_tensor();
}
std::shared_ptr<descriptor::Tensor> Output<Node>::get_tensor_ptr() const {
    return m_node->m_outputs.at(m_index).get_tensor_ptr();
}
const element::Type& Output<Node>::get_element_type() const {
    return m_node->get_output_element_type(m_index);
}
void Output<Node>::set_tensor_ptr(std::shared_ptr<descriptor::Tensor> tensor_ptr) {
    return m_node->m_outputs.at(m_index).set_tensor_ptr(tensor_ptr);
}
const Shape& Output<Node>::get_shape() const {
    return m_node->get_output_shape(m_index);
}
const PartialShape& Output<Node>::get_partial_shape() const {
    return m_node->get_output_partial_shape(m_index);
}

std::set<Input<Node>> Output<Node>::get_target_inputs() const {
    std::set<Input<Node>> result;

    for (auto& input : m_node->m_outputs.at(m_index).get_inputs()) {
        result.emplace(input->get_raw_pointer_node(), input->get_index());
    }

    return result;
}

void Output<Node>::remove_target_input(const Input<Node>& target_input) const {
    m_node->m_outputs.at(m_index).remove_input(&(target_input.get_node()->m_inputs.at(target_input.get_index())));
}

void Output<Node>::replace(const Output<Node>& replacement) {
    for (auto& input : get_target_inputs()) {
        if (input.get_node() != replacement.get_node())
            input.replace_source_output(replacement);
    }
    replacement.get_tensor_ptr()->add_names(get_tensor_ptr()->get_names());
    OPENVINO_SUPPRESS_DEPRECATED_START
    // In legacy API we rely on output port tensor name and use it as an input or output name for the model
    // Due to m_name is just a string, and we can't store multiple aliases for single output port we have to
    // handle two situations during replacement:
    // 1. When we replace consumers to Parameter output port we can't change its name, so we skip this part
    // 2. In other cases when we replace consumers to another output port we should set name. For example:
    //    if we eliminate Node2 from Node1->Node2->Result we have to set Node2 output port name to Node1
    //    output port name, so the output name for model won't be changed.
    // But there are some cases when output name can not be preserved, so the replacement shouldn't be used:
    // 1. Parameter->Node->Result - if we eliminate Node we will lose output name
    // 2. Node1-->Node2->Result - if we eliminate Node2 we will lose Result output name
    //         `->Result
    // In both of these cases please use replace_output_update_name() method which automatically prevents the
    // replacement for cases when we can not preserve input/output names of model.
    if (!is_type<ov::op::v0::Parameter>(replacement.get_node())) {
        ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(),
                                                  ov::descriptor::get_ov_tensor_legacy_name(get_tensor()));
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

    ov::copy_output_runtime_info({*this, replacement}, {replacement});
}

RTMap& Output<Node>::get_rt_info() {
    return m_node->m_outputs.at(m_index).get_rt_info();
}

const RTMap& Output<Node>::get_rt_info() const {
    return m_node->m_outputs.at(m_index).get_rt_info();
}

const RTMap& Output<const Node>::get_rt_info() const {
    return m_node->m_outputs.at(m_index).get_rt_info();
}

const std::unordered_set<std::string>& Output<Node>::get_names() const {
    return m_node->m_outputs.at(m_index).get_tensor_ptr()->get_names();
}

std::string Output<Node>::get_any_name() const {
    return get_tensor().get_any_name();
}

void Output<Node>::set_names(const std::unordered_set<std::string>& names) {
    return m_node->m_outputs.at(m_index).get_tensor_ptr()->set_names(names);
}

void Output<Node>::add_names(const std::unordered_set<std::string>& names) {
    return m_node->m_outputs.at(m_index).get_tensor_ptr()->add_names(names);
}

const std::unordered_set<std::string>& Output<const Node>::get_names() const {
    return m_node->m_outputs.at(m_index).get_tensor_ptr()->get_names();
}

std::string Output<const Node>::get_any_name() const {
    return get_tensor().get_any_name();
}

bool Output<Node>::operator==(const Output& other) const {
    return m_node == other.m_node && m_index == other.m_index;
}
bool Output<Node>::operator!=(const Output& other) const {
    return !(*this == other);
}
bool Output<Node>::operator<(const Output& other) const {
    return m_node->get_instance_id() < other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index < other.m_index);
}
bool Output<Node>::operator>(const Output& other) const {
    return m_node->get_instance_id() > other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index > other.m_index);
}
bool Output<Node>::operator<=(const Output& other) const {
    return !(*this > other);
}
bool Output<Node>::operator>=(const Output& other) const {
    return !(*this < other);
}

Output<Node>::operator Output<const Node>() const {
    return Output<const Node>(get_node(), get_index());
}

Output<const Node>::Output(const Node* node, size_t index) : m_index(index) {
    OPENVINO_ASSERT(node, "Cannot create ov::Output<const ov::Node> from nullptr!");
    m_node = node->shared_from_this();
}

Output<const Node>::Output(const std::shared_ptr<const Node>& node, size_t index) : m_node(node), m_index(index) {}

void Output<const Node>::reset() {
    m_node.reset();
    m_index = 0;
}

Output<const Node> Output<const Node>::for_node(const std::shared_ptr<const Node>& node) {
    return Output(node, m_index);
}

const Node* Output<const Node>::get_node() const {
    return m_node.get();
}
std::shared_ptr<const Node> Output<const Node>::get_node_shared_ptr() const {
    return m_node;
}
size_t Output<const Node>::get_index() const {
    return m_index;
}
descriptor::Tensor& Output<const Node>::get_tensor() const {
    return m_node->m_outputs.at(m_index).get_tensor();
}
std::shared_ptr<descriptor::Tensor> Output<const Node>::get_tensor_ptr() const {
    return m_node->m_outputs.at(m_index).get_tensor_ptr();
}
const element::Type& Output<const Node>::get_element_type() const {
    return m_node->get_output_element_type(m_index);
}
const Shape& Output<const Node>::get_shape() const {
    return m_node->get_output_shape(m_index);
}
const PartialShape& Output<const Node>::get_partial_shape() const {
    return m_node->get_output_partial_shape(m_index);
}

std::set<Input<Node>> Output<const Node>::get_target_inputs() const {
    std::set<Input<Node>> result;

    for (auto& input : m_node->m_outputs.at(m_index).get_inputs()) {
        result.emplace(input->get_raw_pointer_node(), input->get_index());
    }

    return result;
}

bool Output<const Node>::operator==(const Output& other) const {
    return m_node == other.m_node && m_index == other.m_index;
}
bool Output<const Node>::operator!=(const Output& other) const {
    return !(*this == other);
}
bool Output<const Node>::operator<(const Output& other) const {
    return m_node->get_instance_id() < other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index < other.m_index);
}
bool Output<const Node>::operator>(const Output& other) const {
    return m_node->get_instance_id() > other.m_node->get_instance_id() ||
           (m_node == other.m_node && m_index > other.m_index);
}
bool Output<const Node>::operator<=(const Output& other) const {
    return !(*this > other);
}
bool Output<const Node>::operator>=(const Output& other) const {
    return !(*this < other);
}
std::ostream& operator<<(std::ostream& out, const Output<Node>& output) {
    return output.get_node()->write_description(out, 0)
           << "[" << output.get_index() << "]:" << output.get_element_type() << output.get_partial_shape();
}

std::ostream& operator<<(std::ostream& out, const Output<const Node>& output) {
    return output.get_node()->write_description(out, 0)
           << "[" << output.get_index() << "]:" << output.get_element_type() << output.get_partial_shape();
}
}  // namespace ov
