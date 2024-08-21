// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "openvino/frontend/exception.hpp"
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
    return ov::element::undefined;
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
                                     [&place](ov::frontend::Place::Ptr const& output) {
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
