//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <onnx/onnx_pb.h>

#include "ngraph/except.hpp"
#include "onnx_editor/edge_mapper.hpp"

using namespace ngraph;
using namespace ngraph::onnx_editor;

onnx_editor::EdgeMapper::EdgeMapper(const ONNX_NAMESPACE::GraphProto& graph_proto)
{
    int topological_index = 0;
    m_node_inputs.resize(graph_proto.node().size());
    m_node_outputs.resize(graph_proto.node().size());
    for (const auto& node_proto : graph_proto.node())
    {
        for (const auto& out_name : node_proto.output())
        {
            // node output name is unique
            m_node_name_to_index.emplace(out_name, topological_index);
            m_node_outputs[topological_index].push_back(out_name);
            std::cout << "output: " << topological_index << ", " << out_name << "\n";
        }
        for (const auto& in_name : node_proto.input())
        {
            std::cout << "in_name: " << topological_index << ", " << in_name << "\n";
            m_node_inputs[topological_index].push_back(in_name);
        }
        if (!node_proto.name().empty())
        {
            // node name can identify node, but it can be ambiguous
            m_node_name_to_index.emplace(node_proto.name(), topological_index);
            std::cout << "node_name: " << topological_index << ", " << node_proto.name() << "\n";
        }
        ++topological_index;
    }
}

int onnx_editor::EdgeMapper::find_node_index(const std::string& node_name,
                                             const std::string& output_name) const
{
    for (const auto& key : {node_name, output_name})
    {
        if (key.empty())
        {
            continue;
        }
        const auto& index_iter = m_node_name_to_index.find(key);
        if (index_iter != std::end(m_node_name_to_index))
        {
            return index_iter->second;
        }
    }
    throw ngraph_error("Node with name: " + (node_name.empty() ? "not_given" : node_name) +
                       " and output_name: " + (output_name.empty() ? "not_given" : output_name) +
                       " was not found");
};

std::string onnx_editor::EdgeMapper::get_node_output_name(int node_index, int output_index) const
{
    if (node_index >= m_node_outputs.size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           "is out of scope outputs list");
    }
    if (output_index >= m_node_outputs[node_index].size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           " has not output with index: " + std::to_string(output_index));
    }
    const auto output_name = m_node_outputs[node_index][output_index];
    return output_name;
}

std::string onnx_editor::EdgeMapper::get_node_input_name(int node_index, int input_index) const
{
    if (node_index >= m_node_inputs.size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           "is out of scope inputs list");
    }
    if (input_index >= m_node_inputs[node_index].size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           " has not input with index: " + std::to_string(input_index));
    }
    const auto input_name = m_node_inputs[node_index][input_index];
    return input_name;
}

InputEdge onnx_editor::EdgeMapper::to_input_edge(Node node, Input in) const
{
    // identification can be both based on node name and output name
    const auto node_index = find_node_index(node.m_node_name, node.m_output_name);
    if (!in.m_input_name.empty())
    {
        return InputEdge{node_index, in.m_input_name};
    }
    else if (in.m_input_index != -1) // input index is set
    {
        const auto& input_name = get_node_input_name(node_index, in.m_input_index);
        return InputEdge{node_index, input_name};
    }
    else
    {
        throw ngraph_error("Not enough information to determine input edge");
    }
}

OutputEdge onnx_editor::EdgeMapper::to_output_edge(Node node, Output out) const
{
    // identification can be both based on node name and output name
    const auto node_index = find_node_index(node.m_node_name, node.m_output_name);
    if (!out.m_output_name.empty())
    {
        return OutputEdge{node_index, out.m_output_name};
    }
    else if (out.m_output_index != -1) // output index is set
    {
        const auto& output_name = get_node_output_name(node_index, out.m_output_index);
        return OutputEdge{node_index, output_name};
    }
    else
    {
        throw ngraph_error("Not enough information to determine output edge");
    }
}