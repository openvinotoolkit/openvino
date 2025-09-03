// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"

using namespace ov;
using namespace ov::frontend::onnx;

InputModel::InputModel(const std::string& path, const bool enable_mmap, frontend::ExtensionHolder extensions)
    : m_editor{std::make_shared<ONNXModelEditor>(path, enable_mmap, std::move(extensions))} {}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModel::InputModel(const std::wstring& path, const bool enable_mmap, frontend::ExtensionHolder extensions)
    : m_editor{std::make_shared<ONNXModelEditor>(path, enable_mmap, std::move(extensions))} {}
#endif

InputModel::InputModel(std::istream& model_stream, const bool enable_mmap, frontend::ExtensionHolder extensions)
    : m_editor{std::make_shared<ONNXModelEditor>(model_stream, "", enable_mmap, std::move(extensions))} {}

InputModel::InputModel(std::istream& model_stream,
                       const std::string& path,
                       const bool enable_mmap,
                       frontend::ExtensionHolder extensions)
    : m_editor{std::make_shared<ONNXModelEditor>(model_stream, path, enable_mmap, std::move(extensions))} {}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
InputModel::InputModel(std::istream& model_stream,
                       const std::wstring& path,
                       const bool enable_mmap,
                       frontend::ExtensionHolder extensions)
    : InputModel(model_stream, ov::util::wstring_to_string(path), enable_mmap, std::move(extensions)) {}
#endif

InputModel::InputModel(std::shared_ptr<ModelProto> model_proto, frontend::ExtensionHolder extensions)
    : m_editor{std::make_shared<ONNXModelEditor>(model_proto, std::move(extensions))} {}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    const auto& inputs = m_editor->model_inputs();
    std::vector<ov::frontend::Place::Ptr> in_places;
    in_places.reserve(inputs.size());
    for (const auto& input : inputs) {
        in_places.push_back(std::make_shared<PlaceTensor>(input, m_editor));
    }
    return in_places;
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    const auto& outputs = m_editor->model_outputs();
    std::vector<ov::frontend::Place::Ptr> out_places;
    out_places.reserve(outputs.size());
    for (const auto& output : outputs) {
        out_places.push_back(std::make_shared<PlaceTensor>(output, m_editor));
    }
    return out_places;
}

ov::frontend::Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    if (m_editor->is_correct_tensor_name(tensor_name)) {
        return std::make_shared<PlaceTensor>(tensor_name, m_editor);
    }
    return nullptr;
}

ov::frontend::Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    FRONT_END_NOT_IMPLEMENTED(get_place_by_input_index);
}

ov::frontend::Place::Ptr InputModel::get_place_by_operation_name(const std::string& operation_name) const {
    if (m_editor->is_correct_and_unambiguous_node(operation_name)) {
        const auto node_index = m_editor->get_node_index(EditorNode{operation_name});
        EditorNode node{node_index};
        node.m_node_name = operation_name;
        return std::make_shared<PlaceOp>(node, m_editor);
    }
    return nullptr;
}

ov::frontend::Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                                int input_port_index) {
    const auto op = get_place_by_operation_name(operation_name);
    if (op != nullptr) {
        return op->get_input_port(input_port_index);
    }
    return nullptr;
}

ov::frontend::Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                                 int output_port_index) {
    const auto op = get_place_by_operation_name(operation_name);
    if (op != nullptr) {
        return op->get_output_port(output_port_index);
    }
    return nullptr;
}

void InputModel::set_name_for_tensor(const ov::frontend::Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_GENERAL_CHECK(tensor, __FUNCTION__, " expects a pointer to place.");

    const auto onnx_tensor = std::dynamic_pointer_cast<PlaceTensor>(tensor);
    FRONT_END_GENERAL_CHECK(onnx_tensor, __FUNCTION__, " expects a pointer to place of ONNX tensor type.");
    const auto original_name = onnx_tensor->get_names().at(0);
    onnx_tensor->set_name(new_name);

    if (m_additional_tensor_names.count(original_name) > 0) {
        m_additional_tensor_names[new_name] = m_additional_tensor_names[original_name];
        m_additional_tensor_names.erase(original_name);
    }

    if (m_inputs_to_reshape.count(original_name) > 0) {
        m_inputs_to_reshape[new_name] = m_inputs_to_reshape[original_name];
        m_inputs_to_reshape.erase(original_name);
    }
}

void InputModel::set_name_for_operation(const ov::frontend::Place::Ptr& operation, const std::string& new_name) {
    FRONT_END_GENERAL_CHECK(operation, __FUNCTION__, " expects a pointer to place.");

    const auto onnx_operation = std::dynamic_pointer_cast<PlaceOp>(operation);
    FRONT_END_GENERAL_CHECK(onnx_operation, __FUNCTION__, " expects a pointer to place of ONNX operation type.");
    onnx_operation->set_name(new_name);
}

void InputModel::free_name_for_operation(const std::string& name) {
    m_editor->clear_nodes_name(name);
}

void InputModel::set_name_for_dimension(const ov::frontend::Place::Ptr& tensor,
                                        size_t shape_dim_index,
                                        const std::string& dim_name) {
    FRONT_END_GENERAL_CHECK(tensor, __FUNCTION__, " expects a pointer to place.");

    const auto onnx_tensor = std::dynamic_pointer_cast<PlaceTensor>(tensor);
    FRONT_END_GENERAL_CHECK(onnx_tensor, __FUNCTION__, " expects a pointer to place of ONNX tensor type.");
    onnx_tensor->set_name_for_dimension(shape_dim_index, dim_name);
}

void InputModel::add_name_for_tensor(const ov::frontend::Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_GENERAL_CHECK(tensor, __FUNCTION__, " expects a pointer to place.");
    FRONT_END_GENERAL_CHECK(!new_name.empty(), "The additional tensor name cannot be empty.");

    ov::frontend::Place::Ptr tensor_place = tensor;
    const auto input_edge = std::dynamic_pointer_cast<PlaceInputEdge>(tensor);
    if (input_edge) {
        tensor_place = input_edge->get_source_tensor();
    }

    const auto onnx_tensor = std::dynamic_pointer_cast<PlaceTensor>(tensor_place);
    FRONT_END_GENERAL_CHECK(onnx_tensor != nullptr,
                            "Incorrect Place passed to add_name_for_tensor. This method expects a PlaceTensor object "
                            "pointing to the ONNX tensor.");

    auto& names_to_add = m_additional_tensor_names[onnx_tensor->get_names().at(0)];
    names_to_add.insert(new_name);
}

void InputModel::free_name_for_tensor(const std::string&) {
    FRONT_END_THROW("Method free_name_for_tensor is not applicable for ONNX model. ONNX tensor name is an identifier.");
}

void InputModel::set_partial_shape(const ov::frontend::Place::Ptr& place, const ov::PartialShape& shape) {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    std::string input_name;  // name of the model input which should be reshaped
    const auto input_edge = std::dynamic_pointer_cast<PlaceInputEdge>(place);
    if (input_edge) {
        const auto tensor_names = input_edge->get_source_tensor()->get_names();
        OPENVINO_ASSERT(!tensor_names.empty(), "Cannot retrieve input name. Setting new input shape is not possible.");
        input_name = tensor_names[0];
    } else {
        // fallback in case something else than an InputEdge is passed in - try to retrieve its name and reshape
        OPENVINO_ASSERT(!place->get_names().empty(),
                        "Cannot retrieve input name. Setting new input shape is not possible.");
        input_name = place->get_names()[0];
    }

    m_editor->set_input_shapes({{input_name, shape}});

    if (shape.get_min_shape() != shape.get_max_shape())
        m_inputs_to_reshape[input_name] = shape;
}

ov::PartialShape InputModel::get_partial_shape(const ov::frontend::Place::Ptr& place) const {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    std::string tensor_name;  // name of the model input which should be reshaped
    const auto input_edge = std::dynamic_pointer_cast<PlaceInputEdge>(place);
    const auto output_edge = std::dynamic_pointer_cast<PlaceOutputEdge>(place);
    if (input_edge) {
        const auto tensor_names = input_edge->get_source_tensor()->get_names();
        OPENVINO_ASSERT(!tensor_names.empty(),
                        "Cannot retrieve source tensor name for this InputEdge and thus partial shape.");
        tensor_name = tensor_names[0];
    } else if (output_edge) {
        const auto tensor_names = output_edge->get_target_tensor()->get_names();
        OPENVINO_ASSERT(!tensor_names.empty(),
                        "Cannot retrieve target tensor name for this OutputEdge and thus partial shape.");
        tensor_name = tensor_names[0];
    } else {
        tensor_name = place->get_names().at(0);
    }

    return m_editor->get_tensor_shape(tensor_name);
}

void InputModel::set_element_type(const ov::frontend::Place::Ptr& place, const ov::element::Type& type) {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    std::map<std::string, ov::element::Type_t> m;
    m[place->get_names().at(0)] = type;
    m_editor->set_input_types(m);
}

ov::element::Type InputModel::get_element_type(const ov::frontend::Place::Ptr& place) const {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    std::string tensor_name;
    const auto input_edge = std::dynamic_pointer_cast<PlaceInputEdge>(place);
    const auto output_edge = std::dynamic_pointer_cast<PlaceOutputEdge>(place);
    if (input_edge) {
        const auto tensor_names = input_edge->get_source_tensor()->get_names();
        OPENVINO_ASSERT(!tensor_names.empty(),
                        "Cannot retrieve source tensor name for this InputEdge and thus its element type.");
        tensor_name = tensor_names[0];
    } else if (output_edge) {
        const auto tensor_names = output_edge->get_target_tensor()->get_names();
        OPENVINO_ASSERT(!tensor_names.empty(),
                        "Cannot retrieve target tensor name for this OutputEdge and thus its element type.");
        tensor_name = tensor_names[0];
    } else {
        OPENVINO_ASSERT(place->get_names().size() > 0, "Place must have its name.");
        tensor_name = place->get_names().at(0);
    }

    if (place->is_input()) {
        return m_editor->get_input_type(tensor_name);
    }
    // now we can return the concrete element type only for model inputs
    return ov::element::dynamic;
}

std::shared_ptr<Model> InputModel::decode() {
    return m_editor->decode();
}

std::shared_ptr<Model> InputModel::convert() {
    auto converted_model = m_editor->get_function();
    add_tensor_names(converted_model);
    reshape_model_inputs(converted_model);
    return converted_model;
}

// Editor features
bool InputModel::is_correct_place(const ov::frontend::Place::Ptr& place) const {
    if (const auto tensor = std::dynamic_pointer_cast<PlaceTensor>(place)) {
        return m_editor->is_correct_tensor_name(tensor->get_names()[0]);
    }
    if (const auto op = std::dynamic_pointer_cast<PlaceOp>(place)) {
        return m_editor->is_correct_and_unambiguous_node(op->get_editor_node());
    }
    if (const auto input_edge = std::dynamic_pointer_cast<PlaceInputEdge>(place)) {
        if (auto tensor = std::dynamic_pointer_cast<PlaceTensor>(input_edge->get_source_tensor())) {
            return m_editor->is_correct_tensor_name(tensor->get_names()[0]);
        }
    }
    if (const auto output_edge = std::dynamic_pointer_cast<PlaceOutputEdge>(place)) {
        if (auto tensor = std::dynamic_pointer_cast<PlaceTensor>(output_edge->get_target_tensor())) {
            return m_editor->is_correct_tensor_name(tensor->get_names()[0]);
        }
    }
    return false;
}

void InputModel::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    std::vector<Place::Ptr> expected_valid_outputs;
    for (const auto& output : outputs) {
        bool is_correct = is_correct_place(output);
        if (!is_correct)
            OPENVINO_WARN("Name  ",
                          output->get_names().at(0),
                          " of output node is not a correct node name. Ignoring this parameter.");
        else
            expected_valid_outputs.push_back(output);
    }

    extract_subgraph({}, expected_valid_outputs);

    FRONT_END_GENERAL_CHECK(std::all_of(std::begin(expected_valid_outputs),
                                        std::end(expected_valid_outputs),
                                        [](const ov::frontend::Place::Ptr& place) {
                                            return place->is_output();
                                        }),
                            "Not all provided arguments of override_all_outputs are new outputs of the model");

    const auto current_outputs = get_outputs();
    FRONT_END_GENERAL_CHECK(std::all_of(std::begin(current_outputs),
                                        std::end(current_outputs),
                                        [&](const Place::Ptr& current_out) {
                                            return std::find_if(std::begin(expected_valid_outputs),
                                                                std::end(expected_valid_outputs),
                                                                [&](const Place::Ptr& expected_out) {
                                                                    return expected_out->is_equal(current_out);
                                                                }) != std::end(current_outputs);
                                        }),
                            "Some other than expected outputs were created during override_all_outputs");
}

void InputModel::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    std::vector<Place::Ptr> expected_valid_inputs;
    for (const auto& input : inputs) {
        bool is_correct = is_correct_place(input);
        if (!is_correct)
            OPENVINO_WARN("Name  ",
                          input->get_names().at(0),
                          " of input node is not a correct node. Ignoring this parameter.");
        else
            expected_valid_inputs.push_back(input);
    }

    const auto outputs_before_extraction = m_editor->model_outputs();
    extract_subgraph({expected_valid_inputs}, {});

    FRONT_END_GENERAL_CHECK(std::equal(std::begin(outputs_before_extraction),
                                       std::end(outputs_before_extraction),
                                       std::begin(m_editor->model_outputs())),
                            "All outputs should be preserved after override_all_inputs. Provided inputs does "
                            "not satisfy all outputs");

    const auto current_inputs = get_inputs();
    FRONT_END_GENERAL_CHECK(std::all_of(std::begin(current_inputs),
                                        std::end(current_inputs),
                                        [&](const Place::Ptr& current_in) {
                                            return std::find_if(std::begin(expected_valid_inputs),
                                                                std::end(expected_valid_inputs),
                                                                [&](const Place::Ptr& expected_in) {
                                                                    return expected_in->is_equal(current_in);
                                                                }) != std::end(current_inputs);
                                        }),
                            "Some other than expected inputs were created during override_all_inputs");
}

void InputModel::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                  const std::vector<ov::frontend::Place::Ptr>& outputs) {
    std::vector<InputEdge> onnx_inputs = convert_place_to_input_edge(inputs);
    std::vector<OutputEdge> onnx_outputs = convert_place_to_output_edge(outputs);

    m_editor->extract_subgraph(onnx_inputs, onnx_outputs);
}

ov::frontend::Place::Ptr InputModel::add_output(const ov::frontend::Place::Ptr& place) {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    std::string name = place->get_names().at(0);

    const auto& outputs = m_editor->model_outputs();
    const auto& inputs = m_editor->model_inputs();

    auto find_output = std::find(std::begin(outputs), std::end(outputs), name);
    auto find_input = std::find(std::begin(inputs), std::end(inputs), name);

    if (find_input != inputs.end()) {
        return nullptr;
    }

    const auto output_port = place->get_producing_port();

    if (find_output != outputs.end()) {
        return place;
    } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensor>(place)) {
        auto tensor_name = tensor->get_names()[0];
        auto output_edge = m_editor->find_output_edge(tensor_name);
        m_editor->add_output(output_edge);
    } else if (const auto onnx_output_edge = std::dynamic_pointer_cast<PlaceOutputEdge>(output_port)) {
        FRONT_END_GENERAL_CHECK(onnx_output_edge, "Non-onnx output place was passed.");
        m_editor->add_output(onnx_output_edge->get_output_edge());
    } else {
        return nullptr;
    }

    return std::make_shared<PlaceTensor>(name, m_editor);
}

void InputModel::remove_output(const ov::frontend::Place::Ptr& place) {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    std::string name = place->get_names().at(0);
    std::vector<ov::frontend::Place::Ptr> outputs = get_outputs();
    const auto& output_names = m_editor->model_outputs();

    auto find_output = std::find(output_names.begin(), output_names.end(), name);

    if (find_output != output_names.end()) {
        outputs.erase(std::remove_if(outputs.begin(),
                                     outputs.end(),
                                     [&place](const ov::frontend::Place::Ptr& output) {
                                         return output->is_equal(place);
                                     }),
                      outputs.end());

        extract_subgraph({}, {outputs});
    }
}

void InputModel::cut_and_add_new_input(const ov::frontend::Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    if (place->is_input())
        return;

    std::vector<ov::frontend::Place::Ptr> inputs = get_inputs();
    std::vector<ov::frontend::Place::Ptr> outputs = get_outputs();

    const auto edge_place = convert_place_to_input_edge({place});
    const auto edge_outputs = convert_place_to_output_edge(outputs);

    if (!edge_place.empty() && !edge_outputs.empty()) {
        m_editor->extract_subgraph(edge_place, edge_outputs, true);

        // change name for newly created input, it is the last entry in get_inputs()
        if (!new_name_optional.empty()) {
            auto new_inputs = get_inputs();
            m_editor->set_tensor_name(new_inputs.back()->get_names().at(0), new_name_optional);
        }
    }
}

void InputModel::set_tensor_value(const ov::frontend::Place::Ptr& place, const void* value) {
    FRONT_END_GENERAL_CHECK(place, __FUNCTION__, " expects a pointer to place.");

    if (const auto var_place = std::dynamic_pointer_cast<PlaceTensor>(place)) {
        std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> map;

        auto name = place->get_names().at(0);
        auto p_shape = m_editor->get_tensor_shape(name);
        auto el_type = m_editor->get_input_type(name);

        std::shared_ptr<ov::op::v0::Constant> constant =
            ov::op::v0::Constant::create(el_type, p_shape.to_shape(), value);

        constant->set_friendly_name(name);
        map.emplace(name, constant);
        m_editor->set_input_values(map);
    }
}

std::vector<InputEdge> InputModel::convert_place_to_input_edge(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    std::vector<InputEdge> onnx_inputs;
    onnx_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (const auto input_port = std::dynamic_pointer_cast<PlaceInputEdge>(input)) {
            input_port->check_if_valid();
            onnx_inputs.push_back(input_port->get_input_edge());
        } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensor>(input)) {
            const auto name = tensor->get_names().at(0);
            const auto consumers = m_editor->find_output_consumers(name);
            std::transform(std::begin(consumers),
                           std::end(consumers),
                           std::back_inserter(onnx_inputs),
                           [](const InputEdge& edge) {
                               return edge;
                           });
        } else if (const auto op = std::dynamic_pointer_cast<PlaceOp>(input)) {
            op->check_if_valid();
            const auto editor_node = op->get_editor_node();
            const auto op_inputs = m_editor->get_input_ports(editor_node);
            int node_idx = m_editor->get_node_index(editor_node);
            int port_idx = 0;
            std::transform(std::begin(op_inputs),
                           std::end(op_inputs),
                           std::back_inserter(onnx_inputs),
                           [&node_idx, &port_idx](const std::string&) {
                               return InputEdge{node_idx, port_idx++};
                           });
        }
    }

    return onnx_inputs;
}

std::vector<OutputEdge> InputModel::convert_place_to_output_edge(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    std::vector<OutputEdge> onnx_outputs;
    onnx_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        if (const auto output_port = std::dynamic_pointer_cast<PlaceOutputEdge>(output)) {
            output_port->check_if_valid();
            onnx_outputs.push_back(output_port->get_output_edge());
        } else if (const auto tensor = std::dynamic_pointer_cast<PlaceTensor>(output)) {
            const auto output_port = tensor->get_producing_port();
            const auto onnx_output_edge = std::dynamic_pointer_cast<PlaceOutputEdge>(output_port);
            FRONT_END_GENERAL_CHECK(onnx_output_edge,
                                    "Non-onnx output place was passed as extraction subgraph argument");
            onnx_outputs.push_back(onnx_output_edge->get_output_edge());
        } else if (const auto op = std::dynamic_pointer_cast<PlaceOp>(output)) {
            op->check_if_valid();
            const auto editor_node = op->get_editor_node();
            const auto op_outputs = m_editor->get_output_ports(editor_node);
            int node_idx = m_editor->get_node_index(editor_node);
            int port_idx = 0;
            std::transform(std::begin(op_outputs),
                           std::end(op_outputs),
                           std::back_inserter(onnx_outputs),
                           [&node_idx, &port_idx](const std::string&) {
                               return OutputEdge{node_idx, port_idx++};
                           });
        }
    }

    return onnx_outputs;
}

void InputModel::add_tensor_names(std::shared_ptr<Model>& model) {
    FRONT_END_GENERAL_CHECK(model, __FUNCTION__, " expects a pointer to model.");

    auto model_inputs = model->inputs();
    const auto find_input_by_tensor_name = [&model_inputs](const std::string& name) {
        return std::find_if(std::begin(model_inputs),
                            std::end(model_inputs),
                            [&name](const ov::OutputVector::value_type& input) {
                                return input.get_names().count(name) > 0;
                            });
    };

    for (auto& tensor_names : m_additional_tensor_names) {
        auto it = find_input_by_tensor_name(tensor_names.first);
        // add names only to the tensors which still exist in the converted model
        // multiple graph cuts might have removed some parts of the model which initially required additional names
        if (it != model_inputs.end()) {
            it->add_names(tensor_names.second);
        }
    }
}

void InputModel::reshape_model_inputs(std::shared_ptr<Model>& model) {
    FRONT_END_GENERAL_CHECK(model, __FUNCTION__, " expects a pointer to model.");

    const auto& inputs = model->inputs();
    const auto is_input_name = [&inputs](const std::string& name) {
        return std::find_if(std::begin(inputs), std::end(inputs), [&name](const ov::OutputVector::value_type& input) {
                   return input.get_names().count(name) > 0;
               }) != std::end(inputs);
    };

    // assure that names actually refer to model's inputs
    std::map<std::string, ov::PartialShape> actual_inputs_to_reshape;
    for (const auto& in : m_inputs_to_reshape)
        if (is_input_name(in.first)) {
            actual_inputs_to_reshape.insert(in);
        }

    if (!actual_inputs_to_reshape.empty()) {
        model->reshape(actual_inputs_to_reshape);
    }
}

namespace ov {
namespace frontend {
namespace onnx {
namespace unify {

class InputModel::InputModelONNXImpl {
public:
    InputModelONNXImpl(const GraphIterator::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
    InputModelONNXImpl(const GraphIterator::Ptr& graph_iterator,
                       const ov::frontend::InputModel& input_model,
                       const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<ov::frontend::Place::Ptr> get_inputs() const;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const;
    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const;

    /////  Searching for places  /////
    std::vector<std::shared_ptr<OpPlace>> get_op_places() const {
        return m_op_places;
    }
    std::map<std::string, std::shared_ptr<TensorONNXPlace>> get_tensor_places() const {
        return m_tensor_places;
    }
    std::map<std::string, Output<ov::Node>> get_tensor_values() const {
        return m_tensor_values;
    }

    ///// Naming and annotation  /////
    void set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name);
    void add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name);
    void set_name_for_operation(const Place::Ptr& operation, const std::string& new_name);

    ///// Setting / getting tensor properties  /////
    void set_partial_shape(ov::frontend::Place::Ptr place, const ov::PartialShape& shape);
    ov::PartialShape get_partial_shape(ov::frontend::Place::Ptr place) const;
    void set_element_type(ov::frontend::Place::Ptr place, const ov::element::Type& type);
    ov::element::Type get_element_type(ov::frontend::Place::Ptr place) const;
    void set_tensor_value(ov::frontend::Place::Ptr place, const void* value);

    ///// Topology Editing  /////
    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                          const std::vector<ov::frontend::Place::Ptr>& outputs);

    std::vector<std::shared_ptr<ov::frontend::onnx::unify::InputModel>> get_subgraphs();

private:
    void load_model();
    void clean_up();

    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
    std::map<std::string, std::shared_ptr<TensorONNXPlace>> m_tensor_places;
    std::vector<ov::frontend::Place::Ptr> m_inputs;
    std::vector<ov::frontend::Place::Ptr> m_outputs;
    std::map<std::string, Output<ov::Node>> m_tensor_values;

    std::shared_ptr<GraphIterator> m_graph_iterator;
    const ov::frontend::InputModel& m_input_model;
    std::vector<std::shared_ptr<ov::frontend::onnx::unify::InputModel>> m_subgraphs;
    std::shared_ptr<TelemetryExtension> m_telemetry;
};

namespace {
std::shared_ptr<ov::frontend::onnx::TensorONNXPlace> decode_tensor_place(
    const ov::frontend::onnx::TensorMetaInfo& tensor_meta_info,
    const ov::frontend::InputModel& model) {
    auto tensor_place =
        std::make_shared<ov::frontend::onnx::TensorONNXPlace>(model,
                                                              tensor_meta_info.m_partial_shape,
                                                              tensor_meta_info.m_element_type,
                                                              std::vector<std::string>{tensor_meta_info.m_tensor_name},
                                                              tensor_meta_info.m_tensor_data);
    return tensor_place;
}

std::shared_ptr<ov::frontend::onnx::TensorONNXPlace> decode_input_tensor(
    const std::shared_ptr<ov::frontend::onnx::DecoderBaseOperation>& decoder,
    size_t idx,
    const ov::frontend::InputModel& model) {
    const auto& tensor_meta_info = decoder->get_input_tensor_info(idx);
    return decode_tensor_place(tensor_meta_info, model);
}

std::shared_ptr<ov::frontend::onnx::TensorONNXPlace> decode_output_tensor(
    const std::shared_ptr<ov::frontend::onnx::DecoderBaseOperation>& decoder,
    size_t idx,
    const ov::frontend::InputModel& model) {
    const auto& tensor_meta_info = decoder->get_output_tensor_info(idx);
    return decode_tensor_place(tensor_meta_info, model);
}
}  // namespace

void InputModel::InputModelONNXImpl::load_model() {
    std::map<std::string, uint64_t> op_statistics;  // for telemetry

    m_op_places.reserve(m_graph_iterator->size());
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        const auto& decoder = m_graph_iterator->get_decoder();

        if (auto tensor_decoder = std::dynamic_pointer_cast<DecoderBaseTensor>(decoder)) {
            auto tensor_place = decode_tensor_place(tensor_decoder->get_tensor_info(), m_input_model);
            tensor_place->set_input_index(tensor_decoder->get_input_idx());
            tensor_place->set_output_index(tensor_decoder->get_output_idx());
            FRONT_END_GENERAL_CHECK(tensor_place->is_input() || tensor_place->is_output());
            auto name = tensor_place->get_names()[0];
            if (m_tensor_places.count(name) == 0) {
                m_tensor_places[name] = tensor_place;
                if (tensor_place->is_input())
                    m_inputs.push_back(tensor_place);
                if (tensor_place->is_output())
                    m_outputs.push_back(tensor_place);
            }
            continue;
        }
        m_op_places.push_back(std::make_shared<OpPlace>(m_input_model, decoder));

        if (m_telemetry) {
            op_statistics[decoder->get_op_type()]++;
        }

        auto operation_decoder = std::dynamic_pointer_cast<DecoderBaseOperation>(decoder);
        FRONT_END_GENERAL_CHECK(operation_decoder, "Operation decoder is expected");
        for (size_t i = 0; i < operation_decoder->get_input_size(); ++i) {
            auto place = decode_input_tensor(operation_decoder, i, m_input_model);
            auto name = place->get_names()[0];
            if (m_tensor_places.count(name) == 0) {
                m_tensor_places[name] = place;
                if (auto data = place->get_data()) {
                    auto constant = ov::op::v0::Constant::create(place->get_element_type(),
                                                                 place->get_partial_shape().to_shape(),
                                                                 data);
                    constant->set_friendly_name(name);
                    m_tensor_values[name] = constant;
                } else if (place->get_partial_shape() == PartialShape{0}) {  // empty constant
                    auto constant = ov::op::v0::Constant::create(place->get_element_type(),
                                                                 place->get_partial_shape().to_shape(),
                                                                 {});
                    constant->set_friendly_name(name);
                    m_tensor_values[name] = constant;
                } else {
                    FRONT_END_GENERAL_CHECK(false,
                                            "This tensor should be either input, constant or ",
                                            "should be already produced by previous operators: ",
                                            name,
                                            ". Error is encountered while working with operation of type ",
                                            operation_decoder->get_op_type(),
                                            " and name ",
                                            operation_decoder->get_op_name(),
                                            ".");
                }
            }
        }
        for (size_t i = 0; i < operation_decoder->get_output_size(); ++i) {
            auto place = decode_output_tensor(operation_decoder, i, m_input_model);
            auto name = place->get_names()[0];
            if (m_tensor_places.count(name) == 0)
                m_tensor_places[name] = place;
        }
    }

    auto sorting_places_by_idx = [](bool are_input_places) {
        return
            [are_input_places](const ov::frontend::Place::Ptr& lhs_place, const ov::frontend::Place::Ptr& rhs_place) {
                auto tflite_lhs_place = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(lhs_place);
                auto tflite_rhs_place = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(rhs_place);
                FRONT_END_GENERAL_CHECK(tflite_lhs_place != nullptr && tflite_rhs_place != nullptr,
                                        "TFLite Frontend works with TensorONNXPlaces only");
                size_t rhs_idx, lhs_idx;
                if (are_input_places) {
                    lhs_idx = tflite_lhs_place->get_input_index();
                    rhs_idx = tflite_rhs_place->get_input_index();
                } else {
                    lhs_idx = tflite_lhs_place->get_output_index();
                    rhs_idx = tflite_rhs_place->get_output_index();
                }
                return lhs_idx < rhs_idx;
            };
    };
    std::sort(m_inputs.begin(), m_inputs.end(), sorting_places_by_idx(true));
    std::sort(m_outputs.begin(), m_outputs.end(), sorting_places_by_idx(false));

    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "tflite_" + op.first, static_cast<int>(op.second));
        }
    }

    size_t subgraph_size = m_graph_iterator->get_subgraph_size();
    if (subgraph_size > 1) {
        m_subgraphs.reserve(subgraph_size);
        m_subgraphs.push_back(nullptr);  // no main graph
        for (size_t i = 1; i < subgraph_size; ++i) {
            m_subgraphs.push_back(
                std::make_shared<ov::frontend::onnx::unify::InputModel>(m_graph_iterator->get_subgraph(i), m_telemetry));
        }
    }
}

InputModel::InputModelONNXImpl::InputModelONNXImpl(const GraphIterator::Ptr& graph_iterator,
                                                   const ov::frontend::InputModel& input_model)
    : m_graph_iterator(graph_iterator),
      m_input_model(input_model) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    load_model();
}

InputModel::InputModelONNXImpl::InputModelONNXImpl(const GraphIterator::Ptr& graph_iterator,
                                                   const ov::frontend::InputModel& input_model,
                                                   const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_graph_iterator(graph_iterator),
      m_input_model(input_model),
      m_telemetry(telemetry) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    load_model();
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelONNXImpl::get_inputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelONNXImpl::get_outputs() const {
    return m_outputs;
}

std::shared_ptr<TensorPlace> castToTensorPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlace>(place)) {
        return var_place;
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlace.");
}

ov::frontend::Place::Ptr InputModel::InputModelONNXImpl::get_place_by_tensor_name(const std::string& tensorName) const {
    if (m_tensor_places.find(tensorName) != m_tensor_places.end())
        return castToTensorPlace(m_tensor_places.at(tensorName));
    else
        return nullptr;
}

std::shared_ptr<OpPlace> castToOpPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<OpPlace>(place)) {
        return var_place;
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlace.");
}

void InputModel::InputModelONNXImpl::set_partial_shape(ov::frontend::Place::Ptr place, const PartialShape& shape) {
    castToTensorPlace(place)->set_partial_shape(shape);
}

ov::PartialShape InputModel::InputModelONNXImpl::get_partial_shape(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void InputModel::InputModelONNXImpl::set_element_type(ov::frontend::Place::Ptr place, const element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

ov::element::Type InputModel::InputModelONNXImpl::get_element_type(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::InputModelONNXImpl::set_tensor_value(ov::frontend::Place::Ptr place, const void* value) {
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    FRONT_END_GENERAL_CHECK(tensor_place->get_names().size() > 0,
                            "TensorFlow Lite Frontend: place to be frozen must have the name.");
    auto name = tensor_place->get_names()[0];
    FRONT_END_GENERAL_CHECK(p_shape.is_static(),
                            "TensorFlow Lite Frontend: specify static shape for " + name + " to be frozen.");
    FRONT_END_GENERAL_CHECK(type.is_static(),
                            "TensorFlow Lite Frontend: define static size type for " + name + " to be frozen.");
    auto constant = ov::op::v0::Constant::create(type, p_shape.to_shape(), value);
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

void InputModel::InputModelONNXImpl::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    castToTensorPlace(tensor)->set_names({new_name});
}

void InputModel::InputModelONNXImpl::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    auto tf_tensor = castToTensorPlace(tensor);
    auto names = tf_tensor->get_names();
    names.push_back(new_name);
    tf_tensor->set_names(names);
}

void InputModel::InputModelONNXImpl::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    auto op = castToOpPlace(operation);
    auto names = op->get_names();
    names.push_back(new_name);
    op->set_names(names);
}

void InputModel::InputModelONNXImpl::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    for (const auto& input_place : m_inputs) {
        auto input_lite_place = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(input_place);
        FRONT_END_GENERAL_CHECK(input_lite_place != nullptr, "Input Model has unexpected place as input");
        input_lite_place->set_input_index(-1);
    }
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
    clean_up();
}

void InputModel::InputModelONNXImpl::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    for (const auto& output_place : m_outputs) {
        auto output_lite_place = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "Input Model has unexpected place as output");
        output_lite_place->set_output_index(-1);
    }
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
    clean_up();
}

void InputModel::InputModelONNXImpl::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                                      const std::vector<ov::frontend::Place::Ptr>& outputs) {
    for (const auto& output_place : m_outputs) {
        auto output_lite_place = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "Input Model has unexpected place as output");
        output_lite_place->set_output_index(-1);
    }
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
    for (const auto& output_place : m_outputs) {
        auto output_lite_place = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "Input Model has unexpected place as output");
        output_lite_place->set_output_index(-1);
    }
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
    clean_up();
}

void InputModel::InputModelONNXImpl::clean_up() {
    // TODO: remove all the unnecessary tensors and operations. Could be postponed as TF Lite is OOB type of FrontEnd
}

std::vector<std::shared_ptr<ov::frontend::onnx::unify::InputModel>> InputModel::InputModelONNXImpl::get_subgraphs() {
    return m_subgraphs;
}

InputModel::InputModel(const GraphIterator::Ptr& graph_iterator, const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelONNXImpl>(graph_iterator, *this, telemetry)} {}

std::vector<std::shared_ptr<ov::frontend::onnx::OpPlace>> InputModel::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<ov::frontend::onnx::TensorONNXPlace>> InputModel::get_tensor_places() const {
    return _impl->get_tensor_places();
}

std::map<std::string, Output<ov::Node>> InputModel::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return _impl->get_inputs();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return _impl->get_outputs();
}

ov::frontend::Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->get_place_by_tensor_name(tensorName);
}

ov::frontend::Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    FRONT_END_NOT_IMPLEMENTED(get_place_by_input_index);
}

void InputModel::set_partial_shape(const Place::Ptr& place, const PartialShape& shape) {
    _impl->set_partial_shape(place, shape);
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    return _impl->get_partial_shape(place);
}

void InputModel::set_element_type(const Place::Ptr& place, const element::Type& type) {
    _impl->set_element_type(place, type);
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    return _impl->get_element_type(place);
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    _impl->set_tensor_value(place, value);
}

void InputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    _impl->set_name_for_tensor(tensor, new_name);
}

void InputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    _impl->add_name_for_tensor(tensor, new_name);
}

void InputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    _impl->set_name_for_operation(operation, new_name);
}

void InputModel::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->override_all_outputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    _impl->override_all_inputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                  const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->extract_subgraph(inputs, outputs);
}

std::vector<std::shared_ptr<InputModel>> InputModel::get_subgraphs() const {
    return _impl->get_subgraphs();
}

}  // namespace unify
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
