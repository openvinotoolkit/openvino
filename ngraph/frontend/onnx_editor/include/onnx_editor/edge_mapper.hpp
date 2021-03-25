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

#include <map>
#include <string>
#include <vector>

#include "onnx_editor/editor_types.hpp"

namespace ONNX_NAMESPACE
{
    // forward declaration to avoid the necessity of include paths setting in components
    // that don't directly depend on the ONNX library
    class GraphProto;
} // namespace ONNX_NAMESPACE

namespace ngraph
{
    namespace onnx_editor
    {
        /// \brief A class allows to determine InputEdge and OutputEdge by user-friendly onnx names.
        class EdgeMapper
        {
        public:
            EdgeMapper() = delete;

            /// \brief Creates an edge mapper based on graph_proto state.
            ///
            /// \note If state of graph_proto will be changed, the information
            ///       from edge mapper is outdated.
            ///
            /// \param graph_proto Reference to GraphProto object.
            EdgeMapper(const ONNX_NAMESPACE::GraphProto& graph_proto);

            /// \brief Returns the InputEdge based on a node (node name or output name)
            ///        and an input (input name or input index).
            ///
            /// \note  The node name can be ambiguous (many nodes can have the same name).
            ///        In such a case the algorthim try to match the given node name
            ///        with the input name (given input index is not enough).
            ///        If after such operation a found edge is unique, it is returned.
            ///        If InputEdge cannot be determined based on given params the ngraph_error
            ///        exception is thrown.
            ///
            /// \param node A node helper structure created based on a node name
            ///             or a node output name.
            ///
            /// \param input A input helper structure created based on a input name
            ///              or a input index.
            InputEdge to_input_edge(const Node& node, const Input& input) const;

            /// \brief Returns the OutputEdge based on a node (node name or output name)
            ///        and an output (output name or output index).
            ///
            /// \note  The node name can be ambiguous (many nodes can have the same name).
            ///        In such a case the algorthim try to match the given node name
            ///        with the output name (given output index is not enough).
            ///        If after such operation a found edge is unique, it is returned.
            ///        If OutputEdge cannot be determined based on given params the ngraph_error
            ///        exception is thrown.
            ///
            /// \param node A node helper structure created based on a node name
            ///             or a node output name.
            ///
            /// \param output A output helper structure created based on a output name
            ///               or a output index.
            OutputEdge to_output_edge(const Node& node, const Output& output) const;

        private:
            std::vector<int> find_node_indexes(const std::string& node_name,
                                               const std::string& output_name) const;
            std::string get_node_input_name(int node_index, int input_index) const;
            std::string get_node_output_name(int node_index, int output_index) const;

            std::vector<std::vector<std::string>> m_node_inputs;
            std::vector<std::vector<std::string>> m_node_outputs;
            std::multimap<std::string, int> m_node_name_to_index;
        };
    }
}
