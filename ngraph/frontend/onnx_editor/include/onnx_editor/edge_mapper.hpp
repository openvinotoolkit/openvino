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
        class EdgeMapper
        {
        public:
            EdgeMapper() = delete;
            EdgeMapper(const ONNX_NAMESPACE::GraphProto& graph_proto);

            InputEdge to_input_edge(Node node, Input in) const;
            OutputEdge to_output_edge(Node node, Output out) const;

        private:
            int find_node_index(const std::string& node_name, const std::string& output_name) const;
            std::string get_node_input_name(int node_index, int input_index) const;
            std::string get_node_output_name(int node_index, int output_index) const;

            std::vector<std::vector<std::string>> m_node_inputs;
            std::vector<std::vector<std::string>> m_node_outputs;
            std::multimap<std::string, int> m_node_name_to_index;
        };
    }
}