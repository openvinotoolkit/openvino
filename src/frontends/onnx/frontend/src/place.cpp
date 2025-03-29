// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include <openvino/frontend/exception.hpp>

using namespace ov;
using namespace ov::frontend::onnx;

PlaceInputEdge::PlaceInputEdge(const InputEdge& edge, std::shared_ptr<ONNXModelEditor> editor)
    : m_edge{edge},
      m_editor{std::move(editor)},
      m_initial_source_tensor_name{m_editor->get_source_tensor_name(m_edge)} {}

PlaceInputEdge::PlaceInputEdge(InputEdge&& edge, std::shared_ptr<ONNXModelEditor> editor)
    : m_edge{std::move(edge)},
      m_editor{std::move(editor)},
      m_initial_source_tensor_name{m_editor->get_source_tensor_name(m_edge)} {}

void PlaceInputEdge::check_if_valid() const {
    FRONT_END_GENERAL_CHECK(m_editor->get_source_tensor_name(m_edge) == m_initial_source_tensor_name,
                            "The place ",
                            get_names().at(0),
                            " is outdated since the topology of the model has been changed.");
}

InputEdge PlaceInputEdge::get_input_edge() const {
    return m_edge;
}

std::vector<std::string> PlaceInputEdge::get_names() const {
    return {"InputEdge{" + std::to_string(m_edge.m_node_idx) + ", " + std::to_string(m_edge.m_port_idx) + "}"};
}

bool PlaceInputEdge::is_input() const {
    check_if_valid();
    return m_editor->is_input(m_edge);
}

bool PlaceInputEdge::is_output() const {
    return false;
}

bool PlaceInputEdge::is_equal(const Place::Ptr& another) const {
    if (m_editor->get_source_tensor_name(m_edge) != m_initial_source_tensor_name) {
        return false;
    }
    if (const auto in_edge = std::dynamic_pointer_cast<PlaceInputEdge>(another)) {
        const auto& editor_edge = in_edge->get_input_edge();
        return (editor_edge.m_node_idx == m_edge.m_node_idx) && (editor_edge.m_port_idx == m_edge.m_port_idx);
    }
    return false;
}

bool PlaceInputEdge::is_equal_data(const Place::Ptr& another) const {
    return get_source_tensor()->is_equal_data(another);
}

ov::frontend::Place::Ptr PlaceInputEdge::get_source_tensor() const {
    check_if_valid();
    return std::make_shared<PlaceTensor>(m_editor->get_source_tensor_name(m_edge), m_editor);
}

std::vector<ov::frontend::Place::Ptr> PlaceInputEdge::get_consuming_operations() const {
    check_if_valid();
    return {std::make_shared<PlaceOp>(EditorNode{m_edge.m_node_idx}, m_editor)};
}

ov::frontend::Place::Ptr PlaceInputEdge::get_producing_operation() const {
    return get_source_tensor()->get_producing_operation();
}

ov::frontend::Place::Ptr PlaceInputEdge::get_producing_port() const {
    return get_source_tensor()->get_producing_port();
}

PlaceOutputEdge::PlaceOutputEdge(const OutputEdge& edge, std::shared_ptr<ONNXModelEditor> editor)
    : m_edge{edge},
      m_editor{std::move(editor)},
      m_initial_target_tensor_name{m_editor->get_target_tensor_name(edge)} {}

PlaceOutputEdge::PlaceOutputEdge(OutputEdge&& edge, std::shared_ptr<ONNXModelEditor> editor)
    : m_edge{std::move(edge)},
      m_editor{std::move(editor)},
      m_initial_target_tensor_name{m_editor->get_target_tensor_name(m_edge)} {}

void PlaceOutputEdge::check_if_valid() const {
    bool is_valid_place = m_editor->get_target_tensor_name(m_edge) == m_initial_target_tensor_name;
    FRONT_END_GENERAL_CHECK(is_valid_place,
                            "The place ",
                            get_names().at(0),
                            " is outdated since the topology of the model has been changed.");
}

OutputEdge PlaceOutputEdge::get_output_edge() const {
    return m_edge;
}

std::vector<std::string> PlaceOutputEdge::get_names() const {
    return {"OutputEdge{" + std::to_string(m_edge.m_node_idx) + ", " + std::to_string(m_edge.m_port_idx) + "}"};
}

bool PlaceOutputEdge::is_input() const {
    return false;
}

bool PlaceOutputEdge::is_output() const {
    check_if_valid();
    return m_editor->is_output(m_edge);
}

bool PlaceOutputEdge::is_equal(const Place::Ptr& another) const {
    if (m_editor->get_target_tensor_name(m_edge) != m_initial_target_tensor_name) {
        return false;
    }
    if (const auto out_edge = std::dynamic_pointer_cast<PlaceOutputEdge>(another)) {
        const auto& editor_edge = out_edge->get_output_edge();
        return (editor_edge.m_node_idx == m_edge.m_node_idx) && (editor_edge.m_port_idx == m_edge.m_port_idx);
    }
    return false;
}

bool PlaceOutputEdge::is_equal_data(const Place::Ptr& another) const {
    return get_target_tensor()->is_equal_data(another);
}

ov::frontend::Place::Ptr PlaceOutputEdge::get_target_tensor() const {
    check_if_valid();
    return std::make_shared<PlaceTensor>(m_editor->get_target_tensor_name(m_edge), m_editor);
}

std::vector<ov::frontend::Place::Ptr> PlaceOutputEdge::get_consuming_ports() const {
    return get_target_tensor()->get_consuming_ports();
}

ov::frontend::Place::Ptr PlaceOutputEdge::get_producing_operation() const {
    check_if_valid();
    return std::make_shared<PlaceOp>(EditorNode{m_edge.m_node_idx}, m_editor);
}

std::vector<ov::frontend::Place::Ptr> PlaceOutputEdge::get_consuming_operations() const {
    return get_target_tensor()->get_consuming_operations();
}

PlaceTensor::PlaceTensor(const std::string& name, std::shared_ptr<ONNXModelEditor> editor)
    : m_name{name},
      m_editor{std::move(editor)} {}

PlaceTensor::PlaceTensor(std::string&& name, std::shared_ptr<ONNXModelEditor> editor)
    : m_name{std::move(name)},
      m_editor{std::move(editor)} {}

std::vector<std::string> PlaceTensor::get_names() const {
    return {m_name};
}

ov::frontend::Place::Ptr PlaceTensor::get_producing_port() const {
    FRONT_END_GENERAL_CHECK(!is_input(),
                            "Tensor: " + m_name + " is an input of the model and doesn't have producing port.");
    return std::make_shared<PlaceOutputEdge>(m_editor->find_output_edge(m_name), m_editor);
}

std::vector<ov::frontend::Place::Ptr> PlaceTensor::get_consuming_ports() const {
    std::vector<ov::frontend::Place::Ptr> ret;
    auto edges = m_editor->find_output_consumers(m_name);
    std::transform(edges.begin(), edges.end(), std::back_inserter(ret), [this](const InputEdge& edge) {
        return std::make_shared<PlaceInputEdge>(edge, this->m_editor);
    });
    return ret;
}

ov::frontend::Place::Ptr PlaceTensor::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

bool PlaceTensor::is_input() const {
    const auto inputs = m_editor->model_inputs();
    return std::find(std::begin(inputs), std::end(inputs), m_name) != std::end(inputs);
}

bool PlaceTensor::is_output() const {
    const auto outputs = m_editor->model_outputs();
    return std::find(std::begin(outputs), std::end(outputs), m_name) != std::end(outputs);
}

bool PlaceTensor::is_equal(const Place::Ptr& another) const {
    if (const auto tensor = std::dynamic_pointer_cast<PlaceTensor>(another)) {
        return m_name == tensor->get_names().at(0);
    }
    return false;
}

bool PlaceTensor::is_equal_data(const Place::Ptr& another) const {
    const auto consuming_ports = get_consuming_ports();
    const auto eq_to_consuming_port = [&consuming_ports](const Ptr& another) {
        return std::any_of(consuming_ports.begin(), consuming_ports.end(), [&another](const Ptr& place) {
            return place->is_equal(another);
        });
    };
    return is_equal(another) || (is_input() ? false : get_producing_port()->is_equal(another)) ||
           eq_to_consuming_port(another);
}

std::vector<ov::frontend::Place::Ptr> PlaceTensor::get_consuming_operations() const {
    std::vector<ov::frontend::Place::Ptr> consuming_ports = get_consuming_ports();
    std::vector<ov::frontend::Place::Ptr> consuming_ops;
    std::transform(std::begin(consuming_ports),
                   std::end(consuming_ports),
                   std::back_inserter(consuming_ops),
                   [](const Place::Ptr& place) {
                       return place->get_consuming_operations().at(0);
                   });

    return consuming_ops;
}

void PlaceTensor::set_name(const std::string& new_name) {
    if (m_name == new_name)
        return;
    m_editor->set_tensor_name(m_name, new_name);
    m_name = new_name;
}

void PlaceTensor::set_name_for_dimension(size_t shape_dim_index, const std::string& dim_name) {
    m_editor->set_name_for_dimension(m_name, shape_dim_index, dim_name);
}

PlaceOp::PlaceOp(const EditorNode& node, std::shared_ptr<ONNXModelEditor> editor)
    : m_node{node},
      m_editor{std::move(editor)},
      m_initial_first_output{m_editor->get_output_ports(m_node).at(0)} {}

PlaceOp::PlaceOp(EditorNode&& node, std::shared_ptr<ONNXModelEditor> editor)
    : m_node{std::move(node)},
      m_editor{std::move(editor)},
      m_initial_first_output{m_editor->get_output_ports(m_node).at(0)} {}

void PlaceOp::check_if_valid() const {
    FRONT_END_GENERAL_CHECK(m_editor->is_correct_and_unambiguous_node(m_node) &&
                                m_editor->get_output_ports(m_node).at(0) == m_initial_first_output,
                            "The place ",
                            get_names().at(0),
                            " is outdated since the topology of the model has been changed.");
}

std::vector<std::string> PlaceOp::get_names() const {
    if (!m_node.m_node_name.empty()) {
        return {m_node.m_node_name};
    } else {
        return {m_editor->get_node_name(m_node)};
    }
}

const EditorNode& PlaceOp::get_editor_node() const {
    return m_node;
}

ov::frontend::Place::Ptr PlaceOp::get_output_port() const {
    if (m_editor->get_output_ports(m_node).size() == 1) {
        return get_output_port(0);
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_output_port(int output_port_index) const {
    check_if_valid();
    const int out_ports_number = static_cast<int>(m_editor->get_output_ports(m_node).size());
    if (output_port_index < out_ports_number) {
        return std::make_shared<PlaceOutputEdge>(m_editor->find_output_edge(m_node, EditorOutput{output_port_index}),
                                                 m_editor);
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_output_port(const std::string& output_port_name) const {
    check_if_valid();
    const auto output_ports = m_editor->get_output_ports(m_node);
    if (std::count(std::begin(output_ports), std::end(output_ports), output_port_name) == 1) {
        return std::make_shared<PlaceOutputEdge>(m_editor->find_output_edge(m_node, EditorOutput{output_port_name}),
                                                 m_editor);
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_input_port() const {
    if (m_editor->get_input_ports(m_node).size() == 1) {
        return get_input_port(0);
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_input_port(int input_port_index) const {
    check_if_valid();
    const int in_ports_number = static_cast<int>(m_editor->get_input_ports(m_node).size());
    if (input_port_index < in_ports_number) {
        return std::make_shared<PlaceInputEdge>(m_editor->find_input_edge(m_node, EditorInput{input_port_index}),
                                                m_editor);
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_input_port(const std::string& input_name) const {
    check_if_valid();
    const auto input_ports = m_editor->get_input_ports(m_node);
    if (std::count(std::begin(input_ports), std::end(input_ports), input_name) == 1) {
        return std::make_shared<PlaceInputEdge>(m_editor->find_input_edge(m_node, EditorInput{input_name}), m_editor);
    }
    return nullptr;
}

std::vector<ov::frontend::Place::Ptr> PlaceOp::get_consuming_ports() const {
    std::vector<ov::frontend::Place::Ptr> consuming_ports;
    const auto out_ports_number = static_cast<int>(m_editor->get_output_ports(m_node).size());
    for (int out_idx = 0; out_idx < out_ports_number; ++out_idx) {
        auto consuming_ops_out = get_output_port(out_idx)->get_consuming_ports();
        consuming_ports.insert(consuming_ports.end(), consuming_ops_out.begin(), consuming_ops_out.end());
    }
    return consuming_ports;
}

namespace {
std::vector<ov::frontend::Place::Ptr> get_consuming_ops(std::vector<ov::frontend::Place::Ptr> input_ports) {
    std::vector<ov::frontend::Place::Ptr> consuming_ops;
    std::transform(std::begin(input_ports),
                   std::end(input_ports),
                   std::back_inserter(consuming_ops),
                   [](const ov::frontend::Place::Ptr place) {
                       return place->get_consuming_operations().at(0);
                   });

    return consuming_ops;
}
}  // namespace

std::vector<ov::frontend::Place::Ptr> PlaceOp::get_consuming_operations() const {
    std::vector<ov::frontend::Place::Ptr> consuming_ports = get_consuming_ports();
    return get_consuming_ops(consuming_ports);
}

std::vector<ov::frontend::Place::Ptr> PlaceOp::get_consuming_operations(int output_port_index) const {
    std::vector<ov::frontend::Place::Ptr> consuming_ports = get_output_port(output_port_index)->get_consuming_ports();
    return get_consuming_ops(consuming_ports);
}

std::vector<ov::frontend::Place::Ptr> PlaceOp::get_consuming_operations(const std::string& output_port_name) const {
    std::vector<ov::frontend::Place::Ptr> consuming_ports = get_output_port(output_port_name)->get_consuming_ports();
    return get_consuming_ops(consuming_ports);
}

ov::frontend::Place::Ptr PlaceOp::get_producing_operation() const {
    const auto input_port = get_input_port();
    if (input_port != nullptr) {
        return input_port->get_producing_operation();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_producing_operation(int input_port_index) const {
    const auto input_port = get_input_port(input_port_index);
    if (input_port != nullptr) {
        return input_port->get_producing_operation();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_producing_operation(const std::string& input_port_name) const {
    const auto input_port = get_input_port(input_port_name);
    if (input_port != nullptr) {
        return input_port->get_producing_operation();
    }
    return nullptr;
}

bool PlaceOp::is_equal(const Place::Ptr& another) const {
    if (m_editor->is_correct_and_unambiguous_node(m_node) &&
        m_editor->get_output_ports(m_node).at(0) != m_initial_first_output) {
        return false;
    }
    if (const auto place_op = std::dynamic_pointer_cast<PlaceOp>(another)) {
        const auto& another_node = place_op->get_editor_node();
        if (m_editor->is_correct_and_unambiguous_node(m_node) ||
            m_editor->is_correct_and_unambiguous_node(another_node)) {
            return m_editor->get_node_index(m_node) == m_editor->get_node_index(another_node);
        }
    }
    return false;
}

ov::frontend::Place::Ptr PlaceOp::get_target_tensor() const {
    const auto output_port = get_output_port();
    if (output_port != nullptr) {
        return output_port->get_target_tensor();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_target_tensor(int output_port_index) const {
    const auto output_port = get_output_port(output_port_index);
    if (output_port != nullptr) {
        return output_port->get_target_tensor();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_target_tensor(const std::string& output_name) const {
    const auto output_port = get_output_port(output_name);
    if (output_port != nullptr) {
        return output_port->get_target_tensor();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_source_tensor() const {
    const auto input_port = get_input_port();
    if (input_port != nullptr) {
        return input_port->get_source_tensor();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_source_tensor(int input_port_index) const {
    const auto input_port = get_input_port(input_port_index);
    if (input_port != nullptr) {
        return input_port->get_source_tensor();
    }
    return nullptr;
}

ov::frontend::Place::Ptr PlaceOp::get_source_tensor(const std::string& input_name) const {
    const auto input_port = get_input_port(input_name);
    if (input_port != nullptr) {
        return input_port->get_source_tensor();
    }
    return nullptr;
}

bool PlaceOp::is_input() const {
    return false;
}

bool PlaceOp::is_output() const {
    return false;
}

void PlaceOp::set_name(const std::string& new_name) {
    m_editor->set_node_name(m_node, new_name);
    m_node.m_node_name = new_name;
}
