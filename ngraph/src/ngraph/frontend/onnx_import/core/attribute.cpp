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

#include "attribute.hpp"
#include "graph.hpp"
#include "model.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            namespace attribute
            {
                template <>
                Graph get_value(const ONNX_NAMESPACE::AttributeProto& attribute)
                {
                    if (attribute.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)
                    {
                        throw error::attribute::InvalidData{attribute.type()};
                    }
                    return get_graph(attribute.g());
                }

                Graph get_graph(const ONNX_NAMESPACE::GraphProto& graph)
                {
                    ONNX_NAMESPACE::ModelProto model_proto;
                    *(model_proto.mutable_graph()) = graph;
                    // We're creating here a model with unset `opset_import` field. This shouldn't
                    // be a problem, since we add ONNX opset as a default available opset. Moreover
                    // if we encounter a node absent in current available opsets we will try
                    // to add it's domain to available opsets.
                    Model model{model_proto};
                    return Graph{graph, model};
                }
            }
        }

        std::vector<Graph> Attribute::get_graph_array(Model& model) const
        {
            std::vector<Graph> result;
            for (const auto& graph : m_attribute_proto->graphs())
            {
                result.emplace_back(graph, model);
            }
            return result;
        }

        Graph Attribute::get_graph(Model& model) const
        {
            return Graph{m_attribute_proto->g(), model};
        }

    } // namespace onnx_import

} // namespace ngraph
