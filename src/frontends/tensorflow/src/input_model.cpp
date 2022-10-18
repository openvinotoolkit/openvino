// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <fstream>
#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
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

    if (left_part.find_first_not_of("0123456789") == std::string::npos) {
        port_type = "in";
        operation_name = right_part;
        port_index = std::atoi(left_part.c_str());
    } else if (right_part.find_first_not_of("0123456789") == std::string::npos) {
        port_type = "out";
        operation_name = left_part;
        port_index = std::atoi(right_part.c_str());
    } else {
        FRONT_END_GENERAL_CHECK(false, "Incorrect port name specified: " + port_name);
    }
}

class InputModel::InputModelTFImpl {
public:
    InputModelTFImpl(const GraphIterator::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
    InputModelTFImpl(const GraphIterator::Ptr& graph_iterator,
                     const ov::frontend::InputModel& input_model,
                     const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<ov::frontend::Place::Ptr> getInputs() const;
    std::vector<ov::frontend::Place::Ptr> getOutputs() const;
    ov::frontend::Place::Ptr getPlaceByTensorName(const std::string& tensorName) const;
    void overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
    void overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
    void extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                         const std::vector<ov::frontend::Place::Ptr>& outputs);
    void setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape&);
    ov::PartialShape getPartialShape(ov::frontend::Place::Ptr place) const;
    void setElementType(ov::frontend::Place::Ptr place, const ov::element::Type&);
    void setTensorValue(ov::frontend::Place::Ptr place, const void* value);

    std::vector<std::shared_ptr<OpPlace>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlace>> get_tensor_places() const {
        return m_tensor_places;
    }
    std::map<std::string, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    };

private:
    void loadPlaces();
    std::vector<std::shared_ptr<OpPlace>> determine_cut_nodes() const;

    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
    mutable std::map<std::string, std::shared_ptr<TensorPlace>> m_tensor_places;
    std::vector<ov::frontend::Place::Ptr> m_inputs;
    std::vector<ov::frontend::Place::Ptr> m_outputs;
    std::map<std::string, Output<Node>> m_tensor_values;

    std::shared_ptr<GraphIterator> m_graph_iterator;
    const ov::frontend::InputModel& m_input_model;

    std::shared_ptr<TelemetryExtension> m_telemetry;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};

void InputModel::InputModelTFImpl::loadPlaces() {
    std::set<std::string> all_op_names;
    std::set<std::string> op_names_with_consumers;
    std::map<std::string, uint64_t> op_statistics;

    m_inputs.clear();
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        auto node_decoder = m_graph_iterator->get_decoder();
        auto op_name = node_decoder->get_op_name();
        auto op_type = node_decoder->get_op_type();

        if (m_telemetry) {
            op_statistics[op_type]++;
        }

        auto op_place = std::make_shared<OpPlace>(m_input_model, node_decoder);
        all_op_names.insert(op_name);
        m_op_places.push_back(op_place);
        m_op_places_map[op_name] = op_place;
        if (op_type == "Placeholder") {
            auto pshape = node_decoder->get_attribute("shape").as<ov::PartialShape>();
            auto type = node_decoder->get_attribute("dtype").as<ov::element::Type>();
            std::vector<std::string> names = {op_name};
            auto tensor_place = std::make_shared<TensorPlace>(m_input_model, pshape, type, names);
            m_tensor_places[op_name] = tensor_place;
            m_inputs.push_back(tensor_place);
        }
        for (size_t input_port_idx = 0; input_port_idx < node_decoder->get_input_size(); ++input_port_idx) {
            std::string producer_op_name;
            size_t producer_output_port_idx;
            try {
                node_decoder->get_input_node(input_port_idx, producer_op_name, producer_output_port_idx);
                op_names_with_consumers.insert(producer_op_name);
            } catch (const std::exception&) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '" + node_decoder->get_op_name() + "', expected input name: '" +
                                producer_op_name +
                                "', expected input port index: " + std::to_string(producer_output_port_idx));
            }
        }
    }

    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "tf_" + op.first, static_cast<int>(op.second));
        }
    }

    std::set<std::string> op_names_without_consumers;
    std::set_difference(all_op_names.begin(),
                        all_op_names.end(),
                        op_names_with_consumers.begin(),
                        op_names_with_consumers.end(),
                        std::inserter(op_names_without_consumers, op_names_without_consumers.begin()));
    m_graph_iterator->reset();

    m_outputs.clear();
    for (auto& output_name : op_names_without_consumers) {
        std::vector<std::string> output_names = {output_name};
        auto output_place =
            std::make_shared<TensorPlace>(m_input_model, ov::PartialShape({}), ov::element::undefined, output_names);
        m_tensor_places[output_name] = output_place;
        m_outputs.push_back(output_place);
    }
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelTFImpl::get_op_places() const {
    if (m_graph_changed) {
        return determine_cut_nodes();
    }
    return m_op_places;
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelTFImpl::determine_cut_nodes() const {
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

    // the traversing algorithm to compute topologically sorted nodes is taken from topological_sort in
    // core/graph_util.hpp
    while (ops_to_do.size() > 0) {
        auto current_operation_place = ops_to_do.top();
        auto current_operation_decoder = current_operation_place->get_decoder();
        auto current_operation_name = current_operation_decoder->get_op_name();
        if (ops_done.count(current_operation_place) == 0) {
            bool can_add = true;
            auto input_count = current_operation_decoder->get_input_size();
            for (size_t input_port_idx = 0; input_port_idx < input_count; ++input_port_idx) {
                std::string producer_name;
                size_t producer_output_port_idx;
                try {
                    current_operation_decoder->get_input_node(input_port_idx, producer_name, producer_output_port_idx);
                } catch (const std::exception& e) {
                    FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " +
                                    std::to_string(input_port_idx) + " for op '" +
                                    current_operation_decoder->get_op_name() + "', expected input name: '" +
                                    producer_name +
                                    "', expected input port index: " + std::to_string(producer_output_port_idx) + '\n');
                }

                // skip conditional edges for all operators
                if (is_conditional_edge(producer_name)) {
                    continue;
                }

                // is_input is a flag to leave producer operation node or not.
                // this producing node is not left if consumer is pruned by its input port,
                // the producer node is pruned by its output port or the producer becomes new input
                // 1. check if the current node is pruned by its input port
                bool is_input = false;
                std::string input_port_name = std::to_string(input_port_idx) + ":" + current_operation_name;
                if (m_tensor_places.find(input_port_name) != m_tensor_places.end()) {
                    const auto& tensor_place = m_tensor_places[input_port_name];
                    is_input |= tensor_place->is_input();
                }

                // 2. check if the producer node is pruned by its output port
                std::string output_port_name = producer_name + ":" + std::to_string(producer_output_port_idx);
                if (m_tensor_places.find(output_port_name) != m_tensor_places.end()) {
                    const auto& tensor_place = m_tensor_places[output_port_name];
                    is_input |= tensor_place->is_input();
                }

                // 3. check if the current node is an input
                FRONT_END_GENERAL_CHECK(m_op_places_map.count(producer_name),
                                        "There is no operation node with name: " + producer_name);
                const auto& producer_operation_place = m_op_places_map.at(producer_name);
                if (m_tensor_places.find(producer_name) != m_tensor_places.end()) {
                    const auto& tensor_place = m_tensor_places[producer_name];
                    is_input |= tensor_place->is_input();
                }

                if (!is_input && ops_done.count(producer_operation_place) == 0) {
                    can_add = false;
                    ops_to_do.push(producer_operation_place);
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
    : m_input_model(input_model),
      m_graph_iterator(graph_iterator) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    loadPlaces();
}

InputModel::InputModelTFImpl::InputModelTFImpl(const GraphIterator::Ptr& graph_iterator,
                                               const ov::frontend::InputModel& input_model,
                                               const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_input_model(input_model),
      m_graph_iterator(graph_iterator),
      m_telemetry(telemetry) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    loadPlaces();
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFImpl::getInputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFImpl::getOutputs() const {
    return m_outputs;
}

ov::frontend::Place::Ptr InputModel::InputModelTFImpl::getPlaceByTensorName(const std::string& tensorName) const {
    if (m_tensor_places.find(tensorName) != m_tensor_places.end())
        return m_tensor_places.at(tensorName);

    // check that operation node exists for which this place is specified
    std::string operation_name;
    size_t port_idx;
    std::string port_type;
    tensorflow::extract_operation_name_and_port(tensorName, operation_name, port_idx, port_type);
    if (m_op_places_map.find(operation_name) != m_op_places_map.end()) {
        std::vector<std::string> names = {tensorName};
        auto m_var_place =
            std::make_shared<TensorPlace>(m_input_model, ov::PartialShape(), ov::element::undefined, names);
        m_tensor_places[tensorName] = m_var_place;
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

void InputModel::InputModelTFImpl::overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    m_graph_changed = true;
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
}

void InputModel::InputModelTFImpl::overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    m_graph_changed = true;
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
}

void InputModel::InputModelTFImpl::extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                                   const std::vector<ov::frontend::Place::Ptr>& outputs) {
    m_graph_changed = true;
    overrideAllInputs(inputs);
    overrideAllOutputs(outputs);
}

void InputModel::InputModelTFImpl::setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape& p_shape) {
    castToTensorPlace(place)->set_partial_shape(p_shape);
}

ov::PartialShape InputModel::InputModelTFImpl::getPartialShape(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void InputModel::InputModelTFImpl::setElementType(ov::frontend::Place::Ptr place, const ov::element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

void InputModel::InputModelTFImpl::setTensorValue(ov::frontend::Place::Ptr place, const void* value) {
    m_graph_changed = true;
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->get_names()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

InputModel::InputModel(const GraphIterator::Ptr& graph_iterator, const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelTFImpl>(graph_iterator, *this, telemetry)} {}

std::vector<std::shared_ptr<OpPlace>> InputModel::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<TensorPlace>> InputModel::get_tensor_places() const {
    return _impl->get_tensor_places();
}

std::map<std::string, Output<Node>> InputModel::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return _impl->getInputs();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return _impl->getOutputs();
}

ov::frontend::Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->getPlaceByTensorName(tensorName);
}

void InputModel::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->overrideAllOutputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    _impl->overrideAllInputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                  const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->extractSubgraph(inputs, outputs);
}

void InputModel::set_partial_shape(const ov::frontend::Place::Ptr& place, const ov::PartialShape& p_shape) {
    _impl->setPartialShape(place, p_shape);
}

ov::PartialShape InputModel::get_partial_shape(const ov::frontend::Place::Ptr& place) const {
    return _impl->getPartialShape(place);
}

void InputModel::set_element_type(const ov::frontend::Place::Ptr& place, const ov::element::Type& type) {
    _impl->setElementType(place, type);
}

void InputModel::set_tensor_value(const ov::frontend::Place::Ptr& place, const void* value) {
    _impl->setTensorValue(place, value);
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
