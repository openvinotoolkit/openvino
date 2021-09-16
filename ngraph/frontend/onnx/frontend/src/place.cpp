// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include <frontend_manager/frontend_exceptions.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

PlaceInputEdgeONNX::PlaceInputEdgeONNX(const onnx_editor::InputEdge& edge,
                                       std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_edge{edge},
      m_editor{std::move(editor)} {}

PlaceInputEdgeONNX::PlaceInputEdgeONNX(onnx_editor::InputEdge&& edge,
                                       std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_edge{std::move(edge)},
      m_editor{std::move(editor)} {}

onnx_editor::InputEdge PlaceInputEdgeONNX::get_input_edge() const {
    return m_edge;
}

bool PlaceInputEdgeONNX::is_input() const {
    return m_editor->is_input(m_edge);
}

bool PlaceInputEdgeONNX::is_output() const {
    return false;
}

bool PlaceInputEdgeONNX::is_equal(Place::Ptr another) const {
    if (const auto in_edge = std::dynamic_pointer_cast<PlaceInputEdgeONNX>(another)) {
        const auto& editor_edge = in_edge->get_input_edge();
        return (editor_edge.m_node_idx == m_edge.m_node_idx) && (editor_edge.m_port_idx == m_edge.m_port_idx);
    }
    return false;
}

bool PlaceInputEdgeONNX::is_equal_data(Place::Ptr another) const {
    return get_source_tensor()->is_equal_data(another);
}

Place::Ptr PlaceInputEdgeONNX::get_source_tensor() const {
    return std::make_shared<PlaceTensorONNX>(m_editor->get_source_tensor_name(m_edge), m_editor);
}

PlaceOutputEdgeONNX::PlaceOutputEdgeONNX(const onnx_editor::OutputEdge& edge,
                                         std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_edge{edge},
      m_editor{std::move(editor)} {}

PlaceOutputEdgeONNX::PlaceOutputEdgeONNX(onnx_editor::OutputEdge&& edge,
                                         std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_edge{std::move(edge)},
      m_editor{std::move(editor)} {}

onnx_editor::OutputEdge PlaceOutputEdgeONNX::get_output_edge() const {
    return m_edge;
}

bool PlaceOutputEdgeONNX::is_input() const {
    return false;
}

bool PlaceOutputEdgeONNX::is_output() const {
    return m_editor->is_output(m_edge);
}

bool PlaceOutputEdgeONNX::is_equal(Place::Ptr another) const {
    if (const auto out_edge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(another)) {
        const auto& editor_edge = out_edge->get_output_edge();
        return (editor_edge.m_node_idx == m_edge.m_node_idx) && (editor_edge.m_port_idx == m_edge.m_port_idx);
    }
    return false;
}

bool PlaceOutputEdgeONNX::is_equal_data(Place::Ptr another) const {
    return get_target_tensor()->is_equal_data(another);
}

Place::Ptr PlaceOutputEdgeONNX::get_target_tensor() const {
    return std::make_shared<PlaceTensorONNX>(m_editor->get_target_tensor_name(m_edge), m_editor);
}

std::vector<Place::Ptr> PlaceOutputEdgeONNX::get_consuming_ports() const {
    return get_target_tensor()->get_consuming_ports();
}

Place::Ptr PlaceOutputEdgeONNX::get_producing_operation() const {
    return std::make_shared<PlaceOpONNX>(onnx_editor::EditorNode{m_edge.m_node_idx}, m_editor);
}

PlaceTensorONNX::PlaceTensorONNX(const std::string& name, std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_name{name},
      m_editor{std::move(editor)} {}

PlaceTensorONNX::PlaceTensorONNX(std::string&& name, std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_name{std::move(name)},
      m_editor{std::move(editor)} {}

std::vector<std::string> PlaceTensorONNX::get_names() const {
    return {m_name};
}

Place::Ptr PlaceTensorONNX::get_producing_port() const {
    FRONT_END_GENERAL_CHECK(!is_input(),
                            "Tensor: " + m_name + " is an input of the model and doesn't have producing port.");
    return std::make_shared<PlaceOutputEdgeONNX>(m_editor->find_output_edge(m_name), m_editor);
}

std::vector<Place::Ptr> PlaceTensorONNX::get_consuming_ports() const {
    std::vector<Place::Ptr> ret;
    auto edges = m_editor->find_output_consumers(m_name);
    std::transform(edges.begin(), edges.end(), std::back_inserter(ret), [this](const onnx_editor::InputEdge& edge) {
        return std::make_shared<PlaceInputEdgeONNX>(edge, this->m_editor);
    });
    return ret;
}

Place::Ptr PlaceTensorONNX::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

bool PlaceTensorONNX::is_input() const {
    const auto inputs = m_editor->model_inputs();
    return std::find(std::begin(inputs), std::end(inputs), m_name) != std::end(inputs);
}

bool PlaceTensorONNX::is_output() const {
    const auto outputs = m_editor->model_outputs();
    return std::find(std::begin(outputs), std::end(outputs), m_name) != std::end(outputs);
}

bool PlaceTensorONNX::is_equal(Place::Ptr another) const {
    if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(another)) {
        return m_name == tensor->get_names().at(0);
    }
    return false;
}

bool PlaceTensorONNX::is_equal_data(Place::Ptr another) const {
    const auto consuming_ports = get_consuming_ports();
    const auto eq_to_consuming_port = [&consuming_ports](const Ptr& another) {
        return std::any_of(consuming_ports.begin(), consuming_ports.end(), [&another](const Ptr& place) {
            return place->is_equal(another);
        });
    };
    return is_equal(another) || (is_input() ? false : get_producing_port()->is_equal(another)) ||
           eq_to_consuming_port(another);
}

PlaceOpONNX::PlaceOpONNX(const onnx_editor::EditorNode& node, std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_node{node},
      m_editor{std::move(editor)} {}

PlaceOpONNX::PlaceOpONNX(onnx_editor::EditorNode&& node, std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_node{std::move(node)},
      m_editor{std::move(editor)} {}

std::vector<std::string> PlaceOpONNX::get_names() const {
    return {m_node.m_node_name};
}

onnx_editor::EditorNode PlaceOpONNX::get_editor_node() const {
    return m_node;
}

Place::Ptr PlaceOpONNX::get_output_port() const {
    if (m_editor->get_output_ports(m_node).size() == 1) {
        return get_output_port(0);
    }
    return nullptr;
}

Place::Ptr PlaceOpONNX::get_output_port(int output_port_index) const {
    if (output_port_index < m_editor->get_output_ports(m_node).size()) {
        return std::make_shared<PlaceOutputEdgeONNX>(
            m_editor->find_output_edge(m_node, onnx_editor::EditorOutput{output_port_index}),
            m_editor);
    }
    return nullptr;
}

Place::Ptr PlaceOpONNX::get_output_port(const std::string& output_port_name) const {
    const auto output_ports = m_editor->get_output_ports(m_node);
    if (std::count(std::begin(output_ports), std::end(output_ports), output_port_name) == 1) {
        return std::make_shared<PlaceOutputEdgeONNX>(
            m_editor->find_output_edge(m_node, onnx_editor::EditorOutput{output_port_name}),
            m_editor);
    }
    return nullptr;
}

Place::Ptr PlaceOpONNX::get_input_port() const {
    if (m_editor->get_input_ports(m_node).size() == 1) {
        return get_input_port(0);
    }
    return nullptr;
}

Place::Ptr PlaceOpONNX::get_input_port(int input_port_index) const {
    if (input_port_index < m_editor->get_input_ports(m_node).size()) {
        return std::make_shared<PlaceInputEdgeONNX>(
            m_editor->find_input_edge(m_node, onnx_editor::EditorInput{input_port_index}),
            m_editor);
    }
    return nullptr;
}

Place::Ptr PlaceOpONNX::get_input_port(const std::string& input_name) const {
    const auto input_ports = m_editor->get_input_ports(m_node);
    if (std::count(std::begin(input_ports), std::end(input_ports), input_name) == 1) {
        return std::make_shared<PlaceInputEdgeONNX>(
            m_editor->find_input_edge(m_node, onnx_editor::EditorInput{input_name}),
            m_editor);
    }
    return nullptr;
}

std::vector<Place::Ptr> PlaceOpONNX::get_consuming_ports() const {
    std::vector<Place::Ptr> consuming_ports;
    const auto out_ports_size = m_editor->get_output_ports(m_node).size();
    for (int out_idx = 0; out_idx < out_ports_size; ++out_idx) {
        auto consuming_ops_out = get_output_port(out_idx)->get_consuming_ports();
        consuming_ports.insert(consuming_ports.end(), consuming_ops_out.begin(), consuming_ops_out.end());
    }
    return consuming_ports;
}

bool PlaceOpONNX::is_equal(Place::Ptr another) const {
    if (const auto place_op = std::dynamic_pointer_cast<PlaceOpONNX>(another)) {
        const auto& another_node = place_op->get_editor_node();
        if (m_editor->is_correct_and_unambiguous_node(m_node) ||
            m_editor->is_correct_and_unambiguous_node(another_node)) {
            return m_editor->get_node_index(m_node) == m_editor->get_node_index(another_node);
        }
    }
    return false;
}

bool PlaceOpONNX::is_input() const {
    return false;
}

bool PlaceOpONNX::is_output() const {
    return false;
}
