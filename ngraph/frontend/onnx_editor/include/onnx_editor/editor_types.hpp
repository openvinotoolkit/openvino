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
    }
}
