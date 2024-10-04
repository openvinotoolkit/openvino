// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <fstream>
#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/graph_iterator.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
void extract_operation_name_and_port(const std::string& port_name,
                                     std::string& operation_name,
                                     size_t& port_index,
                                     std::string& port_type) {
    constexpr char delimeter[] = ":";
    auto pos = port_name.find(delimeter);
    if (pos == std::string::npos) {
        operation_name = port_name;
        port_type = "none";
        port_index = 0;
        return;
    }

    FRONT_END_GENERAL_CHECK((0 < pos) && (pos + 1 < port_name.length()), "Incorrect port name specified: " + port_name);

    auto left_part = port_name.substr(0, pos);
    auto right_part = port_name.substr(pos + 1, port_name.length() - pos);

    // it gives priority to parsing output ports
    // because pruning is less important than out-of-the-box conversion
    // for OOB conversion, the model refers only output ports
    if (right_part.find_first_not_of("0123456789") == std::string::npos) {
        port_type = "out";
        operation_name = left_part;
        port_index = std::atoi(right_part.c_str());
    } else if (left_part.find_first_not_of("0123456789") == std::string::npos) {
        port_type = "in";
        operation_name = right_part;
        port_index = std::atoi(left_part.c_str());
    } else {
        FRONT_END_GENERAL_CHECK(false, "Incorrect port name specified: " + port_name);
    }
}

class InputModel::InputModelTFImpl {
public:
    InputModelTFImpl(const GraphIterator::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
    InputModelTFImpl(const GraphIterator::Ptr& graph_iterator,
                     const ov::frontend::InputModel& input_model,
                     const std::shared_ptr<TelemetryExtension>& telemetry,
                     const std::shared_ptr<VariablesIndex>& variables_index,
                     const std::shared_ptr<std::map<std::string, std::string>> saved_model_input_names,
                     const std::shared_ptr<std::map<std::string, std::string>> saved_model_output_names,
                     const HashTableKeysValuesMap hash_table_keys_map,
                     const HashTableKeysValuesMap hash_table_values_map,
                     const std::shared_ptr<CheckpointV1Reader> checkpoint_v1_reader,
                     const bool native_format = false);
    std::vector<ov::frontend::Place::Ptr> get_inputs() const;
    std::vector<ov::frontend::Place::Ptr> get_outputs() const;
    ov::frontend::Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const;
    void override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
    void override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
    void extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                          const std::vector<ov::frontend::Place::Ptr>& outputs);
    void set_partial_shape(ov::frontend::Place::Ptr place, const ov::PartialShape&);
    ov::PartialShape get_partial_shape(ov::frontend::Place::Ptr place) const;
    void set_element_type(ov::frontend::Place::Ptr place, const ov::element::Type&);
    ov::element::Type get_element_type(ov::frontend::Place::Ptr place) const;
    void set_tensor_value(ov::frontend::Place::Ptr place, const void* value);

    std::vector<std::shared_ptr<OpPlace>> get_op_places();
    std::map<std::string, std::shared_ptr<TensorPlace>> get_tensor_places() const {
        return m_default_places;
    }
    std::map<std::string, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    };
    std::shared_ptr<InputModel> get_body_input_model(const std::string& body_model_name) const;
    std::vector<std::string> get_input_names() const;
    std::vector<std::string> get_output_names() const;
    std::shared_ptr<VariablesIndex> get_variables_index() const;
    std::shared_ptr<std::map<std::string, std::string>> get_saved_model_input_names() const;
    std::shared_ptr<std::map<std::string, std::string>> get_saved_model_output_names() const;
    HashTableKeysValuesMap get_hash_table_keys_map() const;
    HashTableKeysValuesMap get_hash_table_values_map() const;
    void set_variable(const ov::frontend::Place::Ptr& place, const Variable::Ptr& variable);
    Variable::Ptr get_variable(const ov::frontend::Place::Ptr& place) const;
    std::shared_ptr<CheckpointV1Reader> get_checkpoint_v1_reader() const;

private:
    void load_places();
    std::vector<std::shared_ptr<OpPlace>> topologically_sort_op_nodes();

    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
    mutable std::map<std::string, std::shared_ptr<TensorPlace>> m_default_places;
    std::vector<ov::frontend::Place::Ptr> m_inputs;
    std::vector<ov::frontend::Place::Ptr> m_outputs;
    std::map<std::string, Output<Node>> m_tensor_values;

    std::shared_ptr<GraphIterator> m_graph_iterator;
    const ov::frontend::InputModel& m_input_model;

    std::vector<std::string> m_input_names;
    std::unordered_set<std::string> m_found_inputs;
    std::vector<std::string> m_output_names;

    std::shared_ptr<TelemetryExtension> m_telemetry;

    std::shared_ptr<VariablesIndex> m_variables_index;
    std::shared_ptr<std::map<std::string, std::string>> m_saved_model_input_names;
    std::shared_ptr<std::map<std::string, std::string>> m_saved_model_output_names;
    HashTableKeysValuesMap m_hash_table_keys_map;
    HashTableKeysValuesMap m_hash_table_values_map;
    std::map<ov::frontend::Place::Ptr, Variable::Ptr> m_variables_map;
    std::shared_ptr<CheckpointV1Reader> m_checkpoint_v1_reader;

    bool m_native_format;
    bool m_custom_inputs;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};

void InputModel::InputModelTFImpl::load_places() {
    std::set<std::string> all_op_names;
    std::set<std::string> op_names_with_consumers;
    std::map<std::string, uint64_t> op_statistics;

    m_custom_inputs = false;

    m_inputs.clear();
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        auto node_decoder = m_graph_iterator->get_decoder();
        auto op_name = node_decoder->get_op_name();
        auto op_type = node_decoder->get_op_type();

        if (op_type == "Placeholder" && op_name.rfind("unused_control_flow_input", 0) != std::string::npos) {
            continue;
        }

        if (m_telemetry) {
            op_statistics[op_type]++;
        }

        auto op_place = std::make_shared<OpPlace>(m_input_model, node_decoder);
        all_op_names.insert(op_name);
        m_op_places.push_back(op_place);
        m_op_places_map[op_name] = op_place;

        // compute non-terminating nodes in the graph
        // and put such nodes into op_names_with_consumers
        for (size_t input_port_idx = 0; input_port_idx < node_decoder->get_input_size(); ++input_port_idx) {
            std::string producer_op_name;
            std::string producer_output_port_name;
            size_t producer_output_port_idx;
            try {
                node_decoder->get_input_node(input_port_idx,
                                             producer_op_name,
                                             producer_output_port_name,
                                             producer_output_port_idx);
                if (is_conditional_edge(producer_op_name)) {
                    // exclude "^" mark indicating (execution) conditional dependency
                    // for example, "^sub_op" means dependency on a producer node with a name "sub_op"
                    // if a node has dependent operation nodes and has no data consumers,
                    // this node is not terminating and will not output to the Result node
                    producer_op_name = producer_op_name.substr(1);
                }

                op_names_with_consumers.insert(producer_op_name);
            } catch (const std::exception&) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '" + node_decoder->get_op_name() + "', expected input name: '" +
                                producer_op_name +
                                "', expected input port index: " + std::to_string(producer_output_port_idx));
            }
        }

        // put places for all inputs of a model into m_inputs
        if (op_type == "Placeholder" || op_type == "PlaceholderWithDefault") {
            if (m_input_names.size() > 0 &&
                std::find(m_input_names.begin(), m_input_names.end(), op_name + ":0") == m_input_names.end()) {
                // this is a body graph since it contains non-empty m_input_names
                // such node not included into m_input_names should be skipped
                continue;
            }

            // in case Placeholder we put created TensorPlace to both m_default_places container and m_inputs
            // since they can be used if user does not override them
            // in case PlaceholderWithDefault we put created TensorPlace only to m_default_places container
            // so that we know its shape and type for a case of custom input
            // by default, PlaceholderWithDefault is replaced by Constant with the default value
            auto pshape = ov::PartialShape::dynamic();
            auto shape_any = node_decoder->get_attribute("shape");
            if (shape_any.is<ov::PartialShape>()) {
                // sometimes shape attribute can be absent in the graph
                // so we need to check if Any object is initialized first
                pshape = shape_any.as<ov::PartialShape>();
            } else {
                OPENVINO_DEBUG("TensorFlow Frontend: Placeholder ", op_name, " does not have 'shape' attribute");
            }
            auto output_shapes_any = node_decoder->get_attribute("_output_shapes");
            if (pshape.rank().is_static() && pshape.rank().get_length() == 0 &&
                output_shapes_any.is<std::vector<ov::PartialShape>>()) {
                // we know some cases when Placeholder operation has empty scalar `shape` attribute value
                // and non-empty `_output_shapes` attribute value.
                // `_output_shapes` attribute value turns to be correct in this case
                auto output_shapes = output_shapes_any.as<std::vector<ov::PartialShape>>();
                if (output_shapes.size() == 1 && output_shapes[0].rank().is_static()) {
                    pshape = output_shapes[0];
                    OPENVINO_DEBUG("TensorFlow Frontend: Placeholder ",
                                   op_name,
                                   " has shape from '_output_shapes' attribute.");
                }
            }
            auto dtype_any = node_decoder->get_attribute("dtype");
            auto placeholder_name = node_decoder->get_op_name();
            ov::element::Type type = ov::element::dynamic;
            if (dtype_any.is<ov::element::Type>()) {
                type = dtype_any.as<ov::element::Type>();
            }
            std::string internal_tensor_name = op_name + ":0";
            std::vector<std::string> names{internal_tensor_name};
            auto tensor_place = std::make_shared<TensorPlace>(m_input_model, pshape, type, names, op_name);
            m_default_places[internal_tensor_name] = tensor_place;

            if (op_type == "Placeholder") {
                if (m_saved_model_input_names && (m_saved_model_input_names->size() > 0)) {
                    // if input signature is defined,
                    // found input must present in this signature
                    if (m_saved_model_input_names->find(internal_tensor_name) != m_saved_model_input_names->end()) {
                        m_inputs.push_back(tensor_place);
                    }
                } else {
                    m_inputs.push_back(tensor_place);
                }
            }
        } else if (op_type == "input_arg") {
            if (m_input_names.size() > 0 &&
                std::find(m_input_names.begin(), m_input_names.end(), op_name) == m_input_names.end()) {
                // this is a body graph since it contains non-empty m_input_names
                // such node not included into m_input_names should be skipped
                continue;
            }

            // create a tensor place for the body graph parameter node and save it in the m_inputs
            // it allows to set shapes for the body graph InputModel for its more optimal conversion
            auto param_type = node_decoder->get_attribute("type");
            ov::element::Type type = ov::element::dynamic;
            if (param_type.is<ov::element::Type>()) {
                type = param_type.as<ov::element::Type>();
            }
            auto tensor_place = std::make_shared<TensorPlace>(m_input_model,
                                                              ov::PartialShape::dynamic(),
                                                              type,
                                                              std::vector<std::string>{op_name},
                                                              op_name);
            m_inputs.push_back(tensor_place);
        }
    }

    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "tf_" + op.first, static_cast<int>(op.second));
        }
    }
    m_graph_iterator->reset();
    m_outputs.clear();

    // SavedModel, MetaGraph formats have model signature that provides a concrete list of outputs
    // some output can place among intermediate layers (i.e. it can have its output consumers)
    // so just terminal nodes may not cover the whole list of outputs
    if (m_saved_model_output_names) {
        for (const auto& map_name : *m_saved_model_output_names) {
            const auto& output_internal_tensor_name = map_name.first;
            auto output_place = std::make_shared<TensorPlace>(m_input_model,
                                                              ov::PartialShape({}),
                                                              ov::element::dynamic,
                                                              std::vector<std::string>{output_internal_tensor_name});
            m_default_places[output_internal_tensor_name] = output_place;
            m_outputs.push_back(output_place);
        }
        return;
    }

    auto out_names = m_graph_iterator->get_output_names();
    if (!out_names.size()) {
        // treat terminal nodes as the models outputs for the frozen TF1 format
        std::set<std::string> op_names_without_consumers;
        std::set_difference(all_op_names.begin(),
                            all_op_names.end(),
                            op_names_with_consumers.begin(),
                            op_names_with_consumers.end(),
                            std::inserter(op_names_without_consumers, op_names_without_consumers.begin()));
        for (const auto& output_name : op_names_without_consumers) {
            auto output_place = std::make_shared<TensorPlace>(m_input_model,
                                                              ov::PartialShape({}),
                                                              ov::element::dynamic,
                                                              std::vector<std::string>{output_name + ":0"});
            // TODO: Create tensor places for each ouput port, ticket-129464
            m_default_places[output_name + ":0"] = output_place;
            m_outputs.push_back(output_place);
        }
        return;
    }
    for (const auto& output_name : out_names) {
        auto output_place = std::make_shared<TensorPlace>(m_input_model,
                                                          ov::PartialShape({}),
                                                          ov::element::dynamic,
                                                          std::vector<std::string>{output_name});
        m_default_places[output_name] = output_place;
        m_outputs.push_back(output_place);
    }
}
std::shared_ptr<VariablesIndex> InputModel::InputModelTFImpl::get_variables_index() const {
    return m_variables_index;
}

std::shared_ptr<std::map<std::string, std::string>> InputModel::InputModelTFImpl::get_saved_model_input_names() const {
    return m_saved_model_input_names;
}

std::shared_ptr<std::map<std::string, std::string>> InputModel::InputModelTFImpl::get_saved_model_output_names() const {
    return m_saved_model_output_names;
}

HashTableKeysValuesMap InputModel::InputModelTFImpl::get_hash_table_keys_map() const {
    return m_hash_table_keys_map;
}

HashTableKeysValuesMap InputModel::InputModelTFImpl::get_hash_table_values_map() const {
    return m_hash_table_values_map;
}

void InputModel::InputModelTFImpl::set_variable(const ov::frontend::Place::Ptr& place, const Variable::Ptr& variable) {
    m_variables_map[place] = variable;
}

Variable::Ptr InputModel::InputModelTFImpl::get_variable(const ov::frontend::Place::Ptr& place) const {
    return m_variables_map.count(place) > 0 ? m_variables_map.at(place) : nullptr;
}

std::shared_ptr<CheckpointV1Reader> InputModel::InputModelTFImpl::get_checkpoint_v1_reader() const {
    return m_checkpoint_v1_reader;
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelTFImpl::get_op_places() {
    return topologically_sort_op_nodes();
}

std::vector<std::string> InputModel::InputModelTFImpl::get_input_names() const {
    return m_input_names;
}

std::vector<std::string> InputModel::InputModelTFImpl::get_output_names() const {
    return m_output_names;
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelTFImpl::topologically_sort_op_nodes() {
    std::vector<std::shared_ptr<OpPlace>> topologically_sorted_ops;
    std::stack<std::shared_ptr<OpPlace>> ops_to_do;
    std::unordered_set<std::shared_ptr<OpPlace>> ops_done;

    for (const auto& output_place : m_outputs) {
        FRONT_END_GENERAL_CHECK(output_place->get_names().size() > 0, "TensorPlace must have at least one name.");
        auto output_place_name = output_place->get_names()[0];
        std::string operation_name;
        size_t port_idx;
        std::string port_type;
        tensorflow::extract_operation_name_and_port(output_place_name, operation_name, port_idx, port_type);
        FRONT_END_GENERAL_CHECK(m_op_places_map.count(operation_name),
                                "Custom specified output is incorrect: " + output_place_name);
        auto output_operation_place = m_op_places_map.at(operation_name);
        ops_to_do.push(output_operation_place);
    }

    for (const auto& op_place : m_op_places) {
        auto op_decoder = op_place->get_decoder();
        auto op_name = op_decoder->get_op_name();
        if (op_decoder->get_op_type() == "NextIteration") {
            // walk through all NextIteration nodes and put their producers into ops_to_do
            // this is needed to avoid missed nodes in the body graph of TF1 While operation
            std::string producer_name;
            std::string producer_output_port_name;
            size_t producer_output_port_idx;
            op_decoder->get_input_node(0, producer_name, producer_output_port_name, producer_output_port_idx);
            FRONT_END_GENERAL_CHECK(m_op_places_map.count(producer_name),
                                    "[TensorFlow Frontend] internal error or inconsistent model: producer of "
                                    "NextIteration is not found among operation places " +
                                        producer_name);
            ops_to_do.push(m_op_places_map.at(producer_name));
        } else if (op_decoder->get_op_type() == "LookupTableImport" ||
                   op_decoder->get_op_type() == "LookupTableImportV2") {
            // all LookupTableImport nodes must be preserved in a graph for conversion because
            // they can be terminating nodes and contain input values for HashTable initialization
            FRONT_END_GENERAL_CHECK(m_op_places_map.count(op_name),
                                    "[TensorFlow Frontend] internal error or inconsistent model: LookupTableImport "
                                    "operation is not found among operation places " +
                                        op_name);
            ops_to_do.push(m_op_places_map.at(op_name));
        }
    }

    // the traversing algorithm to compute topologically sorted nodes is taken from topological_sort in
    // core/graph_util.hpp
    while (ops_to_do.size() > 0) {
        auto current_operation_place = ops_to_do.top();
        auto current_operation_decoder = current_operation_place->get_decoder();
        auto current_operation_name = current_operation_decoder->get_op_name();
        if (ops_done.count(current_operation_place) == 0) {
            bool can_add = true;
            auto input_count = current_operation_decoder->get_input_size();
            auto current_operation_type = current_operation_decoder->get_op_type();

            if (current_operation_type == "NextIteration") {
                // break the cycle created by NextIteration
                input_count = 0;
                std::string producer_name;
                std::string producer_output_port_name;
                size_t producer_output_port_idx;
                current_operation_decoder->get_input_node(0,
                                                          producer_name,
                                                          producer_output_port_name,
                                                          producer_output_port_idx);
                current_operation_place->set_next_iteration_back_edge(producer_name, producer_output_port_idx);
            }

            for (size_t input_port_idx = 0; input_port_idx < input_count; ++input_port_idx) {
                std::string producer_name;
                std::string producer_output_port_name;
                size_t producer_output_port_idx;
                try {
                    current_operation_decoder->get_input_node(input_port_idx,
                                                              producer_name,
                                                              producer_output_port_name,
                                                              producer_output_port_idx);
                } catch (const std::exception&) {
                    FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " +
                                    std::to_string(input_port_idx) + " for op '" +
                                    current_operation_decoder->get_op_name() + "', expected input name: '" +
                                    producer_name +
                                    "', expected input port index: " + std::to_string(producer_output_port_idx) + '\n');
                }

                if (is_conditional_edge(producer_name)) {
                    // exclude "^" mark indicating (execution) conditional dependency
                    // for example, "^sub_op" means dependency on a producer node with a name "sub_op"
                    // if a node has dependent operation nodes and has no data consumers,
                    // this node is not terminating and will not output to the Result node
                    producer_name = producer_name.substr(1);
                }

                // is_input is a flag to leave producer operation node or not.
                // this producing node is not left if consumer is pruned by its input port,
                // the producer node is pruned by its output port or the producer becomes new input
                // 1. check if the current node is pruned by its input port
                bool is_input = false;
                std::string input_port_name = std::to_string(input_port_idx) + ":" + current_operation_name;
                if (m_default_places.find(input_port_name) != m_default_places.end()) {
                    const auto& tensor_place = m_default_places[input_port_name];
                    is_input |= tensor_place->is_input();
                    m_found_inputs.insert(input_port_name);
                }

                // 2. check if the producer node is pruned by its output port
                std::string output_port_name = producer_name + ":" + std::to_string(producer_output_port_idx);
                if (m_default_places.find(output_port_name) != m_default_places.end()) {
                    const auto& tensor_place = m_default_places[output_port_name];
                    is_input |= tensor_place->is_input();
                    m_found_inputs.insert(output_port_name);
                }

                // 3. check if the current node is an input
                FRONT_END_GENERAL_CHECK(m_op_places_map.count(producer_name),
                                        "There is no operation node with name: " + producer_name);
                const auto& producer_operation_place = m_op_places_map.at(producer_name);
                if (m_default_places.find(producer_name) != m_default_places.end()) {
                    const auto& tensor_place = m_default_places[producer_name];
                    is_input |= tensor_place->is_input();
                    m_found_inputs.insert(producer_name);
                }

                // in case presence of NextIteration in the graph (or cycle created by other operation),
                // we break the cycle by outputs from the NextIteration operation
                // otherwise, the operations nodes in the cycle will be added to ops_to_do infinitely
                if (!is_input && ops_done.count(producer_operation_place) == 0) {
                    can_add = false;
                    ops_to_do.push(producer_operation_place);
                }
            }

            // Storing information about found inputs.
            // It needs to cover "cutting" a graph, we need to return updated list of inputs
            if (current_operation_type == "Placeholder") {
                for (auto& name : current_operation_place->get_names()) {
                    m_found_inputs.insert(name);
                    // Add unified name if needed
                    if (name.find(':') == std::string::npos) {
                        m_found_inputs.insert(name + ":0");
                    }
                }
            }

            if (can_add) {
                topologically_sorted_ops.push_back(current_operation_place);
                ops_to_do.pop();
                ops_done.insert(current_operation_place);
            }
        } else {
            ops_to_do.pop();
        }
    }

    return topologically_sorted_ops;
}

InputModel::InputModelTFImpl::InputModelTFImpl(const GraphIterator::Ptr& graph_iterator,
                                               const ov::frontend::InputModel& input_model)
    : m_graph_iterator(graph_iterator),
      m_input_model(input_model),
      m_native_format(false) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    load_places();
}

std::shared_ptr<InputModel> InputModel::InputModelTFImpl::get_body_input_model(
    const std::string& body_model_name) const {
    auto body_graph_iterator = m_graph_iterator->get_body_graph_iterator(body_model_name);
    if (!body_graph_iterator) {
        return nullptr;
    }
    return std::make_shared<InputModel>(body_graph_iterator, m_telemetry);
}

InputModel::InputModelTFImpl::InputModelTFImpl(
    const GraphIterator::Ptr& graph_iterator,
    const ov::frontend::InputModel& input_model,
    const std::shared_ptr<TelemetryExtension>& telemetry,
    const std::shared_ptr<VariablesIndex>& variables_index,
    const std::shared_ptr<std::map<std::string, std::string>> saved_model_input_names,
    const std::shared_ptr<std::map<std::string, std::string>> saved_model_output_names,
    const HashTableKeysValuesMap hash_table_keys_map,
    const HashTableKeysValuesMap hash_table_values_map,
    const std::shared_ptr<CheckpointV1Reader> checkpoint_v1_reader,
    const bool native_format)
    : m_graph_iterator(graph_iterator),
      m_input_model(input_model),
      m_telemetry(telemetry),
      m_variables_index(variables_index),
      m_saved_model_input_names(saved_model_input_names),
      m_saved_model_output_names(saved_model_output_names),
      m_hash_table_keys_map(hash_table_keys_map),
      m_hash_table_values_map(hash_table_values_map),
      m_checkpoint_v1_reader(checkpoint_v1_reader),
      m_native_format(native_format) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    m_input_names = graph_iterator->get_input_names();
    m_output_names = graph_iterator->get_output_names();
    load_places();
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFImpl::get_inputs() const {
    if (m_native_format) {
        std::vector<ov::frontend::Place::Ptr> found_inputs;
        if (m_custom_inputs) {
            // When user asks overrding inputs/outputs then some inputs should be
            // excluded for output, depends on results after a call of topologically_sort_op_nodes
            // For example, model has a two inputs, but after cutting by an output one input
            // may be unavailable in path to new output. In such case we need to do not
            // return it as an available input, otherwise it won't be connected with a graph.
            for (auto& input : m_inputs) {
                for (auto& name : input->get_names()) {
                    if (std::find(m_found_inputs.begin(), m_found_inputs.end(), name) != m_found_inputs.end()) {
                        found_inputs.push_back(input);
                        break;
                    }
                }
            }
        } else {
            // Do not return internally used inputs
            for (auto& input : m_inputs) {
                for (auto& name : input->get_names()) {
                    if (name == "saver_filename") {
                        continue;
                    }
                    found_inputs.push_back(input);
                }
            }
        }
        return found_inputs;
    } else {
        return m_inputs;
    }
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFImpl::get_outputs() const {
    return m_outputs;
}

ov::frontend::Place::Ptr InputModel::InputModelTFImpl::get_place_by_tensor_name(const std::string& tensorName) const {
    std::string internal_tensor_name = tensorName;

    // For SavedModel format, an user can work with external input names, namely, without `serving_default_` prefix
    // so we have to map it into the internal name to find the tensor in the tensor pool m_default_places.
    // m_saved_model_input_names contains a map from external name to the internal name with port ':0'
    // for example, `input_mask` maps to `serving_default_input_mask:0`
    if (m_saved_model_input_names) {
        for (const auto& alt_name : *m_saved_model_input_names) {
            if (alt_name.second == tensorName) {
                internal_tensor_name = alt_name.first;
                break;
            }
        }
    }

    if (m_saved_model_output_names.get()) {
        for (const auto& alt_name : *m_saved_model_output_names) {
            if (alt_name.second == tensorName) {
                internal_tensor_name = alt_name.first;
                break;
            }
        }
    }

    if (m_default_places.find(internal_tensor_name) != m_default_places.end()) {
        return m_default_places.at(internal_tensor_name);
    } else if (m_default_places.find(internal_tensor_name + ":0") != m_default_places.end()) {
        auto default_place = m_default_places.at(internal_tensor_name + ":0");
        std::vector<std::string> names = {internal_tensor_name};
        auto new_place = std::make_shared<TensorPlace>(m_input_model,
                                                       default_place->get_partial_shape(),
                                                       default_place->get_element_type(),
                                                       names);
        m_default_places[internal_tensor_name] = new_place;
        return new_place;
    }

    // check that operation node exists for which this place is specified
    std::string operation_name;
    size_t port_idx;
    std::string port_type;
    tensorflow::extract_operation_name_and_port(internal_tensor_name, operation_name, port_idx, port_type);

    if (m_op_places_map.find(operation_name) != m_op_places_map.end()) {
        // new Tensor places must be constructed of dynamic rank and type
        std::vector<std::string> names = {internal_tensor_name};
        auto m_var_place =
            std::make_shared<TensorPlace>(m_input_model, ov::PartialShape::dynamic(), ov::element::dynamic, names);
        m_default_places[internal_tensor_name] = m_var_place;
        return m_var_place;
    }

    return nullptr;
}

std::shared_ptr<TensorPlace> castToTensorPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlace>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlace>(place)) {
        return in_port_place->get_source_tensor_tf();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlace>(place)) {
        return out_port_place->get_target_tensor_tf();
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlaceTF.");
}

void InputModel::InputModelTFImpl::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    m_graph_changed = true;
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }

    if (m_native_format) {
        // Need to read actual outputs
        m_custom_inputs = true;
        m_found_inputs.clear();
        topologically_sort_op_nodes();
    }
}

void InputModel::InputModelTFImpl::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    m_graph_changed = true;
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }

    if (m_native_format) {
        // Need to read actual inputs
        m_custom_inputs = true;
        m_found_inputs.clear();
        topologically_sort_op_nodes();
    }
}

void InputModel::InputModelTFImpl::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                                    const std::vector<ov::frontend::Place::Ptr>& outputs) {
    m_graph_changed = true;
    override_all_inputs(inputs);
    override_all_outputs(outputs);
}

void InputModel::InputModelTFImpl::set_partial_shape(ov::frontend::Place::Ptr place, const ov::PartialShape& p_shape) {
    castToTensorPlace(place)->set_partial_shape(p_shape);
}

ov::PartialShape InputModel::InputModelTFImpl::get_partial_shape(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void InputModel::InputModelTFImpl::set_element_type(ov::frontend::Place::Ptr place, const ov::element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

ov::element::Type InputModel::InputModelTFImpl::get_element_type(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::InputModelTFImpl::set_tensor_value(ov::frontend::Place::Ptr place, const void* value) {
    m_graph_changed = true;
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    FRONT_END_GENERAL_CHECK(tensor_place->get_names().size() > 0,
                            "TensorFlow Frontend: place to be frozen must have the name.");
    auto name = tensor_place->get_names()[0];
    FRONT_END_GENERAL_CHECK(p_shape.is_static(),
                            "TensorFlow Frontend: specify static shape for " + name + " to be frozen.");
    FRONT_END_GENERAL_CHECK(type.is_static(),
                            "TensorFlow Frontend: define static size type for " + name + " to be frozen.");
    auto constant = ov::op::v0::Constant::create(type, p_shape.to_shape(), value);
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

InputModel::InputModel(const GraphIterator::Ptr& graph_iterator,
                       const std::shared_ptr<TelemetryExtension>& telemetry,
                       const std::shared_ptr<VariablesIndex>& variables_index,
                       const std::shared_ptr<std::map<std::string, std::string>> saved_model_input_names,
                       const std::shared_ptr<std::map<std::string, std::string>> saved_model_output_names,
                       const HashTableKeysValuesMap hash_table_keys_map,
                       const HashTableKeysValuesMap hash_table_values_map,
                       const std::shared_ptr<CheckpointV1Reader> checkpoint_v1_reader,
                       const bool native_format)
    : _impl{std::make_shared<InputModelTFImpl>(graph_iterator,
                                               *this,
                                               telemetry,
                                               variables_index,
                                               saved_model_input_names,
                                               saved_model_output_names,
                                               hash_table_keys_map,
                                               hash_table_values_map,
                                               checkpoint_v1_reader,
                                               native_format)} {}

std::shared_ptr<VariablesIndex> InputModel::get_variables_index() {
    return _impl->get_variables_index();
}

std::shared_ptr<std::map<std::string, std::string>> InputModel::get_saved_model_input_names() const {
    return _impl->get_saved_model_input_names();
}

std::shared_ptr<std::map<std::string, std::string>> InputModel::get_saved_model_output_names() const {
    return _impl->get_saved_model_output_names();
}

HashTableKeysValuesMap InputModel::get_hash_table_keys_map() const {
    return _impl->get_hash_table_keys_map();
}

HashTableKeysValuesMap InputModel::get_hash_table_values_map() const {
    return _impl->get_hash_table_values_map();
}

void InputModel::set_variable(const ov::frontend::Place::Ptr& place, const Variable::Ptr& variable) {
    _impl->set_variable(place, variable);
}

Variable::Ptr InputModel::get_variable(const ov::frontend::Place::Ptr& place) const {
    return _impl->get_variable(place);
}

std::shared_ptr<CheckpointV1Reader> InputModel::get_checkpoint_v1_reader() const {
    return _impl->get_checkpoint_v1_reader();
}

std::vector<std::string> InputModel::get_input_names() const {
    return _impl->get_input_names();
}

std::vector<std::string> InputModel::get_output_names() const {
    return _impl->get_output_names();
}

std::vector<std::shared_ptr<OpPlace>> InputModel::get_op_places() const {
    return _impl->get_op_places();
}

std::shared_ptr<InputModel> InputModel::get_body_input_model(const std::string& body_model_name) const {
    return _impl->get_body_input_model(body_model_name);
}

std::map<std::string, std::shared_ptr<TensorPlace>> InputModel::get_tensor_places() const {
    return _impl->get_tensor_places();
}

std::map<std::string, Output<Node>> InputModel::get_tensor_values() const {
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

void InputModel::set_partial_shape(const ov::frontend::Place::Ptr& place, const ov::PartialShape& p_shape) {
    _impl->set_partial_shape(place, p_shape);
}

ov::PartialShape InputModel::get_partial_shape(const ov::frontend::Place::Ptr& place) const {
    return _impl->get_partial_shape(place);
}

void InputModel::set_element_type(const ov::frontend::Place::Ptr& place, const ov::element::Type& type) {
    _impl->set_element_type(place, type);
}

ov::element::Type InputModel::get_element_type(const ov::frontend::Place::Ptr& place) const {
    return _impl->get_element_type(place);
}

void InputModel::set_tensor_value(const ov::frontend::Place::Ptr& place, const void* value) {
    _impl->set_tensor_value(place, value);
}
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
