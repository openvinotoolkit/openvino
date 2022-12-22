// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "edge_mapper.hpp"

#include <onnx/onnx_pb.h>

#include <algorithm>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"

using namespace ov;
using namespace ov::onnx_editor;

onnx_editor::EdgeMapper::EdgeMapper(const ONNX_NAMESPACE::GraphProto& graph_proto)
    : m_node_inputs(graph_proto.node().size()),
      m_node_outputs(graph_proto.node().size()) {
    int topological_index = 0;
    for (const auto& node_proto : graph_proto.node()) {
        for (const auto& out_name : node_proto.output()) {
            // node output name is unique
            m_node_output_name_to_index.emplace(out_name, topological_index);
            m_node_outputs[topological_index].push_back(out_name);
        }
        for (const auto& in_name : node_proto.input()) {
            m_node_inputs[topological_index].push_back(in_name);
            m_output_consumers_index.emplace(in_name, topological_index);
        }
        if (!node_proto.name().empty()) {
            // node name can identify node, but it can be ambiguous
            m_node_name_to_index.emplace(node_proto.name(), topological_index);
        }
        ++topological_index;
    }
}

std::vector<int> onnx_editor::EdgeMapper::find_node_indexes(const std::string& node_name,
                                                            const std::string& output_name) const {
    if (!output_name.empty()) {
        const auto& index_iter = m_node_output_name_to_index.find(output_name);
        if (index_iter != std::end(m_node_output_name_to_index)) {
            return std::vector<int>{index_iter->second};
        }
    }
    std::vector<int> result;
    if (!node_name.empty()) {
        const auto matched_nodes_range = m_node_name_to_index.equal_range(node_name);
        std::transform(matched_nodes_range.first,
                       matched_nodes_range.second,
                       std::back_inserter(result),
                       [](const std::pair<std::string, int>& iter) {
                           return iter.second;
                       });
    }
    return result;
};

int onnx_editor::EdgeMapper::get_node_output_idx(int node_index, const std::string& output_name) const {
    NGRAPH_CHECK(node_index >= 0 && node_index < static_cast<int>(m_node_outputs.size()),
                 "Node with index: ",
                 std::to_string(node_index),
                 "is out of scope outputs list");

    const auto& node_outputs = m_node_outputs[node_index];
    const auto out_port_idx = std::find(std::begin(node_outputs), std::end(node_outputs), output_name);
    if (out_port_idx == std::end(node_outputs)) {
        throw ov::Exception("Node with index: " + std::to_string(node_index) +
                            " has not output with name: " + output_name);
    }
    return static_cast<int>(out_port_idx - std::begin(node_outputs));
}

std::vector<int> onnx_editor::EdgeMapper::get_node_input_indexes(int node_index, const std::string& input_name) const {
    NGRAPH_CHECK(node_index >= 0 && node_index < static_cast<int>(m_node_inputs.size()),
                 "Node with index: ",
                 std::to_string(node_index),
                 "is out of scope outputs list");

    const auto& node_inputs = m_node_inputs[node_index];
    std::vector<int> node_inputs_indexes;
    int index = 0;
    for (const auto& in : node_inputs) {
        if (in == input_name) {
            node_inputs_indexes.push_back(index);
        }
        ++index;
    }
    if (node_inputs_indexes.size() == 0) {
        throw ov::Exception("Node with index: " + std::to_string(node_index) +
                            " has not input with name: " + input_name);
    }
    return node_inputs_indexes;
}

InputEdge onnx_editor::EdgeMapper::find_input_edge(const EditorNode& node, const EditorInput& in) const {
    int node_index = node.m_node_index;
    if (node_index == -1) {  // the node index is not provided
        // identification can be both based on node name and output name (if the node index is not provided)
        const auto& node_indexes = find_node_indexes(node.m_node_name, node.m_output_name);
        if (node_indexes.size() == 1) {
            node_index = node_indexes[0];
        } else if (node_indexes.empty()) {
            throw ov::Exception("Node with name: " + (node.m_node_name.empty() ? "not_given" : node.m_node_name) +
                                " and output_name: " + (node.m_output_name.empty() ? "not_given" : node.m_output_name) +
                                " was not found");
        } else if (!in.m_input_name.empty())  // input indexes are not deterministic if a node name is ambiguous
        {
            // many nodes with the same name
            // check if some of found index matches input name
            int matched_inputs_number = 0;
            for (const auto& index : node_indexes) {
                if (std::count(std::begin(m_node_inputs[index]), std::end(m_node_inputs[index]), in.m_input_name) > 0) {
                    node_index = index;
                    ++matched_inputs_number;
                }
            }
            if (matched_inputs_number == 0) {
                throw ov::Exception("Input edge described by: " + node.m_node_name +
                                    " and input name: " + in.m_input_name + " was not found");
            }
            if (matched_inputs_number > 1) {
                throw ov::Exception("Given node name: " + node.m_node_name + " and input name: " + in.m_input_name +
                                    " are ambiguous to determine input edge");
            }
        } else {
            throw ov::Exception("Given node name: " + node.m_node_name + " and input index: " +
                                std::to_string(in.m_input_index) + " are ambiguous to determine input edge");
        }
    } else {  // the node index is provided
        check_node_index(node_index);
    }
    if (in.m_input_index != -1)  // input index is set
    {
        return InputEdge{node_index, in.m_input_index, in.m_new_input_name};
    }
    if (!in.m_input_name.empty()) {
        const auto input_indexes = get_node_input_indexes(node_index, in.m_input_name);
        if (input_indexes.size() > 1)  // more indexes with the same name
        {
            throw ov::Exception("Node with index: " + std::to_string(node_index) +
                                " has more than one inputs with name: " + in.m_input_name +
                                ". You should use port indexes to distinguish them.");
        }
        return InputEdge{node_index, input_indexes[0], in.m_new_input_name};
    } else {
        throw ov::Exception("Not enough information to determine input edge");
    }
}

OutputEdge onnx_editor::EdgeMapper::find_output_edge(const EditorNode& node, const EditorOutput& out) const {
    int node_index = node.m_node_index;
    if (node_index == -1) {  // the node index is not provided
        // identification can be both based on node name and output name (if the node index is not provided)
        const auto& node_indexes = find_node_indexes(node.m_node_name, node.m_output_name);
        if (node_indexes.size() == 1) {
            node_index = node_indexes[0];
        } else if (node_indexes.empty()) {
            throw ov::Exception("Node with name: " + (node.m_node_name.empty() ? "not_given" : node.m_node_name) +
                                " and output_name: " + (node.m_output_name.empty() ? "not_given" : node.m_output_name) +
                                " was not found");
        } else if (!out.m_output_name.empty())  // output indexes are not deterministic if a node name is ambiguous
        {
            // many nodes with the same name
            // check if some of found index matches output name
            int matched_outputs_number = 0;
            for (const auto& index : node_indexes) {
                if (std::count(std::begin(m_node_outputs[index]), std::end(m_node_outputs[index]), out.m_output_name) >
                    0) {
                    node_index = index;
                    ++matched_outputs_number;
                }
            }
            if (matched_outputs_number == 0) {
                throw ov::Exception("Output edge described by: " + node.m_node_name +
                                    " and output name: " + out.m_output_name + " was not found");
            }
        } else {
            throw ov::Exception("Given node name: " + node.m_node_name + " and output index: " +
                                std::to_string(out.m_output_index) + " are ambiguous to determine output edge");
        }
    } else {  // the node index is provided
        check_node_index(node_index);
    }
    if (out.m_output_index != -1)  // output index is set
    {
        return OutputEdge{node_index, out.m_output_index};
    }
    if (!out.m_output_name.empty()) {
        const auto output_idx = get_node_output_idx(node_index, out.m_output_name);
        return OutputEdge{node_index, output_idx};
    } else {
        throw ov::Exception("Not enough information to determine output edge");
    }
}

OutputEdge onnx_editor::EdgeMapper::find_output_edge(const std::string& output_name) const {
    return find_output_edge(EditorNode{EditorOutput{output_name}}, EditorOutput{output_name});
}

std::vector<InputEdge> onnx_editor::EdgeMapper::find_output_consumers(const std::string& output_name) const {
    const auto matched_nodes_range = m_output_consumers_index.equal_range(output_name);
    std::vector<InputEdge> input_edges;
    for (auto it = matched_nodes_range.first; it != matched_nodes_range.second; ++it) {
        const auto node_idx = it->second;
        const auto port_indexes = get_node_input_indexes(node_idx, output_name);
        for (const auto& idx : port_indexes) {
            const auto consumer_edge = InputEdge{node_idx, idx, output_name};
            if (std::find_if(std::begin(input_edges), std::end(input_edges), [&consumer_edge](const InputEdge& edge) {
                    return edge.m_node_idx == consumer_edge.m_node_idx && edge.m_port_idx == consumer_edge.m_port_idx;
                }) == std::end(input_edges)) {
                // only unique
                input_edges.push_back(consumer_edge);
            }
        }
    }
    return input_edges;
}

bool onnx_editor::EdgeMapper::is_correct_and_unambiguous_node(const EditorNode& node) const {
    if (node.m_node_index >= 0 && node.m_node_index < static_cast<int>(m_node_inputs.size())) {
        return true;
    }
    return find_node_indexes(node.m_node_name, node.m_output_name).size() == 1;
}

namespace {
void check_node(bool condition, const EditorNode& node) {
    NGRAPH_CHECK(condition,
                 "The node with name: " + (node.m_node_name.empty() ? "not_given" : node.m_node_name) +
                     ", output_name: " + (node.m_output_name.empty() ? "not_given" : node.m_output_name) +
                     ", node_index: " + (node.m_node_index == -1 ? "not_given" : std::to_string(node.m_node_index)) +
                     " is ambiguous");
}
}  // namespace

int onnx_editor::EdgeMapper::get_node_index(const EditorNode& node) const {
    if (node.m_node_index != -1) {  // the node index provided
        check_node_index(node.m_node_index);
        return node.m_node_index;
    }
    const auto indexes = find_node_indexes(node.m_node_name, node.m_output_name);
    check_node(indexes.size() == 1, node);
    return indexes[0];
}

bool onnx_editor::EdgeMapper::is_correct_tensor_name(const std::string& name) const {
    if (m_node_output_name_to_index.find(name) != std::end(m_node_output_name_to_index)) {
        return true;
    }
    if (m_output_consumers_index.find(name) != std::end(m_output_consumers_index)) {
        return true;
    }
    return false;
}

std::vector<std::string> onnx_editor::EdgeMapper::get_input_ports(const EditorNode& node) const {
    check_node(is_correct_and_unambiguous_node(node), node);
    auto node_index = node.m_node_index;
    if (node_index == -1) {  // the node index is provided
        node_index = find_node_indexes(node.m_node_name, node.m_output_name)[0];
    } else {
        check_node_index(node_index);
    }
    return m_node_inputs[node_index];
}

std::vector<std::string> onnx_editor::EdgeMapper::get_output_ports(const EditorNode& node) const {
    check_node(is_correct_and_unambiguous_node(node), node);
    auto node_index = node.m_node_index;
    if (node_index == -1)  // the node index is provided
    {
        node_index = find_node_indexes(node.m_node_name, node.m_output_name)[0];
    } else {
        check_node_index(node_index);
    }
    return m_node_outputs[node_index];
}

std::string onnx_editor::EdgeMapper::get_source_tensor_name(const InputEdge& edge) const {
    if (edge.m_node_idx >= 0 && edge.m_node_idx < static_cast<int>(m_node_inputs.size()) && edge.m_port_idx >= 0 &&
        edge.m_port_idx < static_cast<int>(m_node_inputs[edge.m_node_idx].size())) {
        return m_node_inputs[edge.m_node_idx][edge.m_port_idx];
    }
    return "";
}

std::string onnx_editor::EdgeMapper::get_target_tensor_name(const OutputEdge& edge) const {
    if (edge.m_node_idx >= 0 && edge.m_node_idx < static_cast<int>(m_node_outputs.size()) && edge.m_port_idx >= 0 &&
        edge.m_port_idx < static_cast<int>(m_node_outputs[edge.m_node_idx].size())) {
        return m_node_outputs[edge.m_node_idx][edge.m_port_idx];
    }
    return "";
}

void onnx_editor::EdgeMapper::check_node_index(int node_index) const {
    NGRAPH_CHECK(node_index >= 0 && node_index < static_cast<int>(m_node_inputs.size()),
                 "Provided node index: " + std::to_string(node_index) + " is out of scope");
}
