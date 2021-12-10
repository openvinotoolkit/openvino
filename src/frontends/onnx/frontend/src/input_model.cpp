// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <common/frontend_exceptions.hpp>
#include <openvino/util/file_util.hpp>

#include "place.hpp"

using namespace ov;
using namespace ov::frontend;

NGRAPH_SUPPRESS_DEPRECATED_START

InputModelONNX::InputModelONNX(const std::string& path,
                               const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(path, telemetry)} {}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModelONNX::InputModelONNX(const std::wstring& path,
                               const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(path, telemetry)} {}
#endif

InputModelONNX::InputModelONNX(std::istream& model_stream,
                               const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(model_stream, "", telemetry)} {}

InputModelONNX::InputModelONNX(std::istream& model_stream,
                               const std::string& path,
                               const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(model_stream, path, telemetry)} {}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
InputModelONNX::InputModelONNX(std::istream& model_stream,
                               const std::wstring& path,
                               const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry)
    : InputModelONNX(model_stream, ov::util::wstring_to_string(path), telemetry) {}
#endif

std::vector<Place::Ptr> InputModelONNX::get_inputs() const {
    const auto& inputs = m_editor->model_inputs();
    std::vector<Place::Ptr> in_places;
    in_places.reserve(inputs.size());
    for (const auto& input : inputs) {
        in_places.push_back(std::make_shared<PlaceTensorONNX>(input, m_editor));
    }
    return in_places;
}

std::vector<Place::Ptr> InputModelONNX::get_outputs() const {
    const auto& outputs = m_editor->model_outputs();
    std::vector<Place::Ptr> out_places;
    out_places.reserve(outputs.size());
    for (const auto& output : outputs) {
        out_places.push_back(std::make_shared<PlaceTensorONNX>(output, m_editor));
    }
    return out_places;
}

Place::Ptr InputModelONNX::get_place_by_tensor_name(const std::string& tensor_name) const {
    if (m_editor->is_correct_tensor_name(tensor_name)) {
        return std::make_shared<PlaceTensorONNX>(tensor_name, m_editor);
    }
    return nullptr;
}

Place::Ptr InputModelONNX::get_place_by_operation_name(const std::string& operation_name) const {
    if (m_editor->is_correct_and_unambiguous_node(operation_name)) {
        const auto node_index = m_editor->get_node_index(onnx_editor::EditorNode{operation_name});
        return std::make_shared<PlaceOpONNX>(onnx_editor::EditorNode{node_index}, m_editor);
    }
    return nullptr;
}

Place::Ptr InputModelONNX::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                      int input_port_index) {
    const auto op = get_place_by_operation_name(operation_name);
    if (op != nullptr) {
        return op->get_input_port(input_port_index);
    }
    return nullptr;
}

Place::Ptr InputModelONNX::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                       int output_port_index) {
    const auto op = get_place_by_operation_name(operation_name);
    if (op != nullptr) {
        return op->get_output_port(output_port_index);
    }
    return nullptr;
}

void InputModelONNX::set_name_for_tensor(Place::Ptr tensor, const std::string& new_name) {
    const auto onnx_tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(tensor);
    FRONT_END_GENERAL_CHECK(onnx_tensor, __FUNCTION__, " expects a pointer to place of ONNX tensor type.");
    onnx_tensor->set_name(new_name);
}

void InputModelONNX::set_name_for_operation(Place::Ptr operation, const std::string& new_name) {
    const auto onnx_operation = std::dynamic_pointer_cast<PlaceOpONNX>(operation);
    FRONT_END_GENERAL_CHECK(onnx_operation, __FUNCTION__, " expects a pointer to place of ONNX operation type.");
    onnx_operation->set_name(new_name);
}

void InputModelONNX::free_name_for_operation(const std::string& name) {
    m_editor->clear_nodes_name(name);
}

void InputModelONNX::set_name_for_dimension(Place::Ptr tensor, size_t shape_dim_index, const std::string& dim_name) {
    const auto onnx_tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(tensor);
    FRONT_END_GENERAL_CHECK(onnx_tensor, __FUNCTION__, " expects a pointer to place of ONNX tensor type.");
    onnx_tensor->set_name_for_dimension(shape_dim_index, dim_name);
}

void InputModelONNX::add_name_for_tensor(Place::Ptr, const std::string&) {
    FRONT_END_THROW("Method add_name_for_tensor is not applicable for ONNX model. ONNX tensor has just one name.");
}

void InputModelONNX::free_name_for_tensor(const std::string&) {
    FRONT_END_THROW("Method free_name_for_tensor is not applicable for ONNX model. ONNX tensor name is an identifier.");
}

void InputModelONNX::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) {
    std::map<std::string, ngraph::PartialShape> m;
    m[place->get_names()[0]] = shape;
    m_editor->set_input_shapes(m);
}

ngraph::PartialShape InputModelONNX::get_partial_shape(Place::Ptr place) const {
    return m_editor->get_tensor_shape(place->get_names().at(0));
}

void InputModelONNX::set_element_type(Place::Ptr place, const ngraph::element::Type& type) {
    std::map<std::string, ngraph::element::Type_t> m;
    m[place->get_names()[0]] = type;
    m_editor->set_input_types(m);
}

std::shared_ptr<Model> InputModelONNX::decode() {
    return m_editor->decode();
}

std::shared_ptr<Model> InputModelONNX::convert() {
    return m_editor->get_function();
}

// Editor features
void InputModelONNX::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    extract_subgraph({}, outputs);
    NGRAPH_CHECK(m_editor->model_outputs().size() == outputs.size(),
                 "Unexpected number of outputs after override_all_outputs");
    NGRAPH_CHECK(std::all_of(std::begin(outputs),
                             std::end(outputs),
                             [](const Place::Ptr& place) {
                                 return place->is_output();
                             }),
                 "Not all provided arguments of override_all_outputs are new outputs of the model");
}

void InputModelONNX::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    const auto outputs_before_extraction = m_editor->model_outputs();
    extract_subgraph({inputs}, {});
    NGRAPH_CHECK(std::equal(std::begin(outputs_before_extraction),
                            std::end(outputs_before_extraction),
                            std::begin(m_editor->model_outputs())),
                 "All outputs should be preserved after override_all_inputs. Provided inputs does "
                 "not satisfy all outputs");
    NGRAPH_CHECK(m_editor->model_inputs().size() == inputs.size(),
                 "Unexpected number of inputs after override_all_inputs");
}

void InputModelONNX::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    std::vector<onnx_editor::InputEdge> onnx_inputs;
    onnx_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (const auto input_port = std::dynamic_pointer_cast<PlaceInputEdgeONNX>(input)) {
            onnx_inputs.push_back(input_port->get_input_edge());
        } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(input)) {
            auto name = tensor->get_names()[0];
            const auto consumers = m_editor->find_output_consumers(name);
            std::transform(std::begin(consumers),
                           std::end(consumers),
                           std::back_inserter(onnx_inputs),
                           [](const onnx_editor::InputEdge& edge) {
                               return edge;
                           });
        } else if (const auto op = std::dynamic_pointer_cast<PlaceOpONNX>(input)) {
            const auto editor_node = op->get_editor_node();
            const auto op_inputs = m_editor->get_input_ports(editor_node);
            int node_idx = m_editor->get_node_index(editor_node);
            int port_idx = 0;
            std::transform(std::begin(op_inputs),
                           std::end(op_inputs),
                           std::back_inserter(onnx_inputs),
                           [&node_idx, &port_idx](const std::string&) {
                               return onnx_editor::InputEdge{node_idx, port_idx++};
                           });
        }
    }

    std::vector<onnx_editor::OutputEdge> onnx_outputs;
    onnx_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        if (const auto output_port = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(output)) {
            onnx_outputs.push_back(output_port->get_output_edge());
        } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensorONNX>(output)) {
            const auto output_port = tensor->get_producing_port();
            const auto onnx_output_edge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(output_port);
            NGRAPH_CHECK(onnx_output_edge, "Non-onnx output place was passed as extraction subgraph argument");
            onnx_outputs.push_back(onnx_output_edge->get_output_edge());
        } else if (const auto op = std::dynamic_pointer_cast<PlaceOpONNX>(output)) {
            const auto editor_node = op->get_editor_node();
            const auto op_outputs = m_editor->get_output_ports(editor_node);
            int node_idx = m_editor->get_node_index(editor_node);
            int port_idx = 0;
            std::transform(std::begin(op_outputs),
                           std::end(op_outputs),
                           std::back_inserter(onnx_outputs),
                           [&node_idx, &port_idx](const std::string&) {
                               return onnx_editor::OutputEdge{node_idx, port_idx++};
                           });
        }
    }
    m_editor->cut_graph_fragment(onnx_inputs, onnx_outputs);
}
