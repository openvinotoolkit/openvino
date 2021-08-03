// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"
#include <frontend_manager/frontend_exceptions.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

PlaceInputEdgeONNX::PlaceInputEdgeONNX(const onnx_editor::InputEdge& edge,
                                       std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_edge{edge}
    , m_editor{editor}
{
}

onnx_editor::InputEdge PlaceInputEdgeONNX::get_input_edge() const
{
    return m_edge;
}

bool PlaceInputEdgeONNX::is_input() const
{
    return m_editor->is_input(m_edge);
}

bool PlaceInputEdgeONNX::is_output() const
{
    return false;
}

bool PlaceInputEdgeONNX::is_equal(Place::Ptr another) const
{
    if (const auto in_edge = std::dynamic_pointer_cast<PlaceInputEdgeONNX>(another))
    {
        const auto& editor_edge = in_edge->get_input_edge();
        return (editor_edge.m_node_idx == m_edge.m_node_idx) &&
               (editor_edge.m_port_idx == m_edge.m_port_idx);
    }
    return false;
}

PlaceOutputEdgeONNX::PlaceOutputEdgeONNX(const onnx_editor::OutputEdge& edge,
                                         std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_edge{edge}
    , m_editor{editor}
{
}

onnx_editor::OutputEdge PlaceOutputEdgeONNX::get_output_edge() const
{
    return m_edge;
}

bool PlaceOutputEdgeONNX::is_input() const
{
    return false;
}

bool PlaceOutputEdgeONNX::is_output() const
{
    return m_editor->is_output(m_edge);
}

bool PlaceOutputEdgeONNX::is_equal(Place::Ptr another) const
{
    if (const auto out_edge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(another))
    {
        const auto& editor_edge = out_edge->get_output_edge();
        return (editor_edge.m_node_idx == m_edge.m_node_idx) &&
               (editor_edge.m_port_idx == m_edge.m_port_idx);
    }
    return false;
}

PlaceTensorONNX::PlaceTensorONNX(const std::string& name,
                                 std::shared_ptr<onnx_editor::ONNXModelEditor> editor)
    : m_name(name)
    , m_editor(editor)
{
}

std::vector<std::string> PlaceTensorONNX::get_names() const
{
    return {m_name};
}

Place::Ptr PlaceTensorONNX::get_producing_port() const
{
    return std::make_shared<PlaceOutputEdgeONNX>(m_editor->find_output_edge(m_name), m_editor);
}

std::vector<Place::Ptr> PlaceTensorONNX::get_consuming_ports() const
{
    std::vector<Place::Ptr> ret;
    auto edges = m_editor->find_output_consumers(m_name);
    std::transform(edges.begin(),
                   edges.end(),
                   std::back_inserter(ret),
                   [this](const onnx_editor::InputEdge& edge) {
                       return std::make_shared<PlaceInputEdgeONNX>(edge, this->m_editor);
                   });
    return ret;
}

Place::Ptr PlaceTensorONNX::get_input_port(int input_port_index) const
{
    return std::make_shared<PlaceInputEdgeONNX>(
        m_editor->find_input_edge(onnx_editor::EditorOutput(m_name),
                                  onnx_editor::EditorInput(input_port_index)),
        m_editor);
}

bool PlaceTensorONNX::is_input() const
{
    const auto inputs = m_editor->model_inputs();
    return std::find(std::begin(inputs), std::end(inputs), m_name) != std::end(inputs);
}

bool PlaceTensorONNX::is_output() const
{
    const auto outputs = m_editor->model_outputs();
    return std::find(std::begin(outputs), std::end(outputs), m_name) != std::end(outputs);
}

bool PlaceTensorONNX::is_equal(Place::Ptr another) const
{
    if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(another))
    {
        return m_name == tensor->get_names().at(0);
    }
    return false;
}
