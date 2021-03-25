// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/attribute.hpp"
#include "core/graph.hpp"
#include "core/model.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::vector<Graph> Attribute::get_graph_array(Model& model) const
        {
            std::vector<Graph> result;
            for (const auto& graph : m_attribute_proto->graphs())
            {
                result.emplace_back(graph, model);
            }
            return result;
        }

        Subgraph Attribute::get_subgraph(const Graph& parent_graph) const
        {
            if (m_attribute_proto->type() != ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)
            {
                throw error::attribute::InvalidData{m_attribute_proto->type()};
            }

            ONNX_NAMESPACE::ModelProto model_proto;
            const auto& graph = m_attribute_proto->g();
            *(model_proto.mutable_graph()) = graph;

            // set opset version and domain from the parent graph
            *model_proto.mutable_opset_import() = parent_graph.get_opset_imports();

            Model model{model_proto};
            return Subgraph{graph, model, parent_graph};
        }

    } // namespace onnx_import

} // namespace ngraph
