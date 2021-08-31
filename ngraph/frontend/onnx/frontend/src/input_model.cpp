// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <frontend_manager/frontend_exceptions.hpp>
#include <ngraph/file_util.hpp>

#include "place.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

NGRAPH_SUPPRESS_DEPRECATED_START

InputModelONNX::InputModelONNX(const std::string& path)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(path)} {}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModelONNX::InputModelONNX(const std::wstring& path)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(path)} {}
#endif

InputModelONNX::InputModelONNX(std::istream& model_stream)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(model_stream)} {}

InputModelONNX::InputModelONNX(std::istream& model_stream, const std::string& path)
    : m_editor{std::make_shared<onnx_editor::ONNXModelEditor>(model_stream, path)} {}

InputModelONNX::InputModelONNX(std::istream& model_stream, const std::wstring& path)
    : InputModelONNX(model_stream, file_util::wstring_to_string(path)) {}

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
    NGRAPH_CHECK(m_editor->is_correct_tensor_name(tensor_name),
                 "The tensor with name: " + tensor_name + " does not exist in the graph");
    return std::make_shared<PlaceTensorONNX>(tensor_name, m_editor);
}

Place::Ptr InputModelONNX::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                      int input_port_index) {
    const auto edge = m_editor->find_input_edge(onnx_editor::EditorNode(operation_name), input_port_index);
    return std::make_shared<PlaceInputEdgeONNX>(edge, m_editor);
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

std::shared_ptr<Function> InputModelONNX::decode() {
    return m_editor->decode();
}

std::shared_ptr<Function> InputModelONNX::convert() {
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
        }
    }

    std::vector<onnx_editor::OutputEdge> onnx_outputs;
    onnx_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        const auto output_port = output->get_producing_port();
        const auto onnx_output_edge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(output_port);
        NGRAPH_CHECK(onnx_output_edge, "Non-onnx output place was passed as extraction subgraph argument");
        onnx_outputs.push_back(onnx_output_edge->get_output_edge());
    }
    m_editor->cut_graph_fragment(onnx_inputs, onnx_outputs);
}
