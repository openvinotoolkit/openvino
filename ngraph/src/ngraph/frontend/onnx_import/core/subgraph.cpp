//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "subgraph.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        SubGraph::SubGraph(const ONNX_NAMESPACE::GraphProto& proto, Model& model, const Graph& parent_graph)
            : Graph(proto, model)
            , m_parent_graph{&parent_graph}
        {
        }

        std::shared_ptr<ngraph::Node> SubGraph::get_ng_node_from_cache(const std::string& name) const
        {
            if(is_node_in_cache(name))
            {
                std::cout << "From subgraph: " << name << "\n";
                return Graph::get_ng_node_from_cache(name);
            }
            else
            {
                std::cout << "From parent graph: " << name << "\n";
                m_parent_graph->get_ng_node_from_cache(name);
            }
            
        }

    } // namespace onnx_import

} // namespace ngraph
