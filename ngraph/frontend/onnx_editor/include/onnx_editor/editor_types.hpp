// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

namespace ngraph
{
    enum class EdgeType
    {
        INPUT,
        OUTPUT
    };

    template <EdgeType>
    struct Edge
    {
        Edge() = delete;
        Edge(const int node_idx, std::string tensor_name)
            : m_node_idx{node_idx}
            , m_tensor_name{std::move(tensor_name)}
        {
        }

        const int m_node_idx;
        const std::string m_tensor_name;
    };
    namespace onnx_editor
    {
        /// \brief Defines an edge connected to an input of any node in the graph.
        ///        It consists of a node index in the processed ONNX model and the input name.
        ///        The index should point to a node in the topological sort of the underlying graph
        ///        which means it has to be in range:  0 <= node_idx < graph.node_size()
        ///
        ///        For a node number 5, with 3 inputs:
        ///
        ///            ----(in_A)---->  +--------+
        ///            ----(in_B)---->  | node 5 |  ----(out)---->
        ///            ----(in_C)---->  +--------+
        ///
        ///        there are 3 possible valid instances of this struct:
        ///            InputEdge(5, "in_A")
        ///            InputEdge(5, "in_B")
        ///            InputEdge(5, "in_C")
        using InputEdge = Edge<EdgeType::INPUT>;

        /// \brief Defines an edge connected to an output of any node in the graph.
        ///        It consists of a node index in the processed ONNX model and the output name.
        ///
        ///        For a node number 5, with 2 outputs:
        ///
        ///                             +--------+  ----(out1)---->
        ///            ----(in_A)---->  | node 5 |
        ///                             +--------+  ----(out2)---->
        ///
        ///        there are 2 possible valid instances of this struct:
        ///            OutputEdge(5, "out1")
        ///            OutputEdge(5, "out2")
        using OutputEdge = Edge<EdgeType::OUTPUT>;

        /// \brief Defines single node input by the name or the index.
        ///        For a node number test_node, with 3 inputs:
        ///
        ///            ----(in_A)---->  +-----------+
        ///            ----(in_B)---->  | test_node |  ----(out)---->
        ///            ----(in_C)---->  +-----------+
        ///        You can indicate in_B as Input("in_B") and Input(1)
        struct Input
        {
            Input() = delete;
            Input(std::string input_name)
                : m_input_name{std::move(input_name)}
            {
            }
            Input(int input_index)
                : m_input_index{std::move(input_index)}
            {
            }
            const std::string m_input_name = "";
            const int m_input_index = -1;
        };

        /// \brief Defines single node output by the name or the index.
        ///        For a node number test_node, with 2 outputs:
        ///
        ///                             +-----------+  ---(out1)--->
        ///            ----(in_A)---->  | test_node |
        ///                             +-----------+  ---(out2)--->
        ///        You can indicate out2 as Output("out2") and Output(1)
        struct Output
        {
            Output() = delete;
            Output(std::string output_name)
                : m_output_name{std::move(output_name)}
            {
            }
            Output(int output_index)
                : m_output_index{std::move(output_index)}
            {
            }
            const std::string m_output_name = "";
            const int m_output_index = -1;
        };

        /// \brief Defines single node by output name which is determinitic
        ///        and node name which can be ambiguous.
        ///        For a node number test_node, with 2 outputs:
        ///
        ///                             +-----------+  ---(out1)--->
        ///            ----(in_A)---->  | test_node |
        ///                             +-----------+  ---(out2)--->
        ///        You can indicate test_node by the name as Node("test_node")
        ///        or by assigned output as Node(Output("out1")) or Node(Output("out2"))
        struct Node
        {
            Node(std::string node_name)
                : m_node_name{std::move(node_name)}
            {
            }
            Node(Output output)
                : m_output_name{output.m_output_name}
            {
            }
            const std::string m_node_name = "";
            const std::string m_output_name = "";
        };

        // Aliases to avoid name conflicts with classes from ngraph namespace
        using EditorInput = Input;
        using EditorOutput = Output;
        using EditorNode = Node;
    }
}
