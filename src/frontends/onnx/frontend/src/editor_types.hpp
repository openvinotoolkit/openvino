// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

namespace ov {
enum class EdgeType { INPUT, OUTPUT };

template <EdgeType>
struct Edge {
    Edge() = delete;
    Edge(const int node_idx, const int port_idx, std::string new_input_name = "")
        : m_node_idx{node_idx},
          m_port_idx{port_idx},
          m_new_input_name{std::move(new_input_name)} {}

    const int m_node_idx;
    const int m_port_idx;
    const std::string m_new_input_name;
};
namespace frontend {
namespace onnx {
/// \brief Defines an edge connected to an input of any node in the graph.
///        It consists of a node index in the processed ONNX model and the port index.
///        The node index should point to a node in the topological sort of the underlying
///        graph which means it has to be in range:  0 <= node_idx < graph.node_size()
///
///        For a node number 5, with 3 inputs:
///
///            ----(0)---->  +--------+
///            ----(1)---->  | node 5 |  ----(0)---->
///            ----(2)---->  +--------+
///
///        there are 3 possible valid instances of this struct:
///            InputEdge(5, 0)
///            InputEdge(5, 1)
///            InputEdge(5, 2)
///
///        If a new_input_name argument is provided, it is used as a new input name
///        in a place where a graph is cut (if creation of a new input is needed).
///        Otherwise, a new input name is set to:
///            - original name of an input tensor, if the tensor is consumed by only a one
///            node
///            - first output name of an input tensor consumer + "/placeholder_port_" +
///            port_index,
///              if the tensor is consumed by more than one node.
using InputEdge = Edge<EdgeType::INPUT>;

/// \brief Defines an edge connected to an output of any node in the graph.
///        It consists of a node index in the processed ONNX model and the port index.
///
///        For a node number 5, with 2 outputs:
///
///                          +--------+  ----(0)---->
///            ----(0)---->  | node 5 |
///                          +--------+  ----(1)---->
///
///        there are 2 possible valid instances of this struct:
///            OutputEdge(5, 0)
///            OutputEdge(5, 1)
///
///        The optional argument "new_input_name" is ignored for OutputEdge case.
using OutputEdge = Edge<EdgeType::OUTPUT>;

/// \brief Specifies a single node input by the name or index.
///
///        For a node test_node, with 3 inputs:
///
///            ----(in_A)---->  +-----------+
///            ----(in_B)---->  | test_node |  ----(out)---->
///            ----(in_C)---->  +-----------+
///        You can indicate in_B as EditorInput("in_B") or EditorInput(1)
///
///        The optional argument "new_input_name" can be used to set a custom input name
///        which can be created during cutting a graph.
struct EditorInput {
    EditorInput() = delete;
    EditorInput(std::string input_name, std::string new_input_name = "")
        : m_input_name{std::move(input_name)},
          m_new_input_name{std::move(new_input_name)} {}
    EditorInput(const int input_index, std::string new_input_name = "")
        : m_input_index{input_index},
          m_new_input_name{std::move(new_input_name)} {}
    const std::string m_input_name = "";
    const int m_input_index = -1;
    const std::string m_new_input_name = "";
};

/// \brief Specifies a single node output by the name or index.
///        For a node test_node, with 2 outputs:
///
///                             +-----------+  ---(out1)--->
///            ----(in_A)---->  | test_node |
///                             +-----------+  ---(out2)--->
///        You can indicate out2 as EditorOutput("out2") or EditorOutput(1)
struct EditorOutput {
    EditorOutput() = delete;
    EditorOutput(std::string output_name) : m_output_name{std::move(output_name)} {}
    EditorOutput(const int output_index) : m_output_index{output_index} {}
    const std::string m_output_name = "";
    const int m_output_index = -1;
};

/// \brief Specifies a single node by output name which is determinitic
///        or node name which can be ambiguous.
///        For a node test_node, with 2 outputs:
///
///                             +-----------+  ---(out1)--->
///            ----(in_A)---->  | test_node |
///                             +-----------+  ---(out2)--->
///        You can indicate test_node by name as EditorNode("test_node")
///        or by assigned output as EditorNode(EditorOutput("out1"))
///        or EditorNode(EditorOutput("out2"))
///        or you can determine the node by postition of a node in an ONNX graph (in topological order).
struct EditorNode {
    EditorNode(std::string node_name) : m_node_name{std::move(node_name)} {}
    EditorNode(EditorOutput output) : m_output_name{std::move(output.m_output_name)} {}
    EditorNode(const int node_index) : m_node_index{node_index} {}
    std::string m_node_name = "";
    std::string m_output_name = "";
    int m_node_index = -1;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
