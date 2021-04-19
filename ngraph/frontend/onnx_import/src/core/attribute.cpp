// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/attribute.hpp"
#include "core/graph.hpp"
#include "core/model.hpp"
#include "ngraph/log.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        Subgraph Attribute::get_subgraph(
            const Graph& parent_graph,
            const std::map<std::string, std::string> parent_subgraph_inputs_map) const
        {
            if (m_attribute_proto->type() != ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)
            {
                throw error::attribute::InvalidData{m_attribute_proto->type()};
            }

            auto model_proto = common::make_unique<ONNX_NAMESPACE::ModelProto>();

            const auto& graph = m_attribute_proto->g();
            model_proto->mutable_graph()->CopyFrom(graph);

            const auto& parent_graph_cache = parent_graph.get_graph_cache();
            auto* subgraph_inputs = model_proto->mutable_graph()->mutable_input();

            // Use the `parent_subgraph_inputs_map` to infer the types for the subgraph inputs
            for (const auto& inputs_pair : parent_subgraph_inputs_map)
            {
                auto subgraph_in = std::find_if(subgraph_inputs->begin(),
                                                subgraph_inputs->end(),
                                                [&](const ONNX_NAMESPACE::ValueInfoProto& in) {
                                                    return in.name() == inputs_pair.second;
                                                });
                if (subgraph_in == subgraph_inputs->end())
                {
                    NGRAPH_WARN << "Input '" << inputs_pair.second
                                << "' was not found in the subgraph for the ONNX Loop operator";
                }
                else
                {
                    const auto& ng_type_from_parent_graph =
                        parent_graph_cache.get_node(inputs_pair.first).get_element_type();
                    subgraph_in->mutable_type()->mutable_tensor_type()->set_elem_type(
                        onnx_common::ng_to_onnx_data_type(ng_type_from_parent_graph));
                }
            }

            // set opset version and domain from the parent graph
            *model_proto->mutable_opset_import() = parent_graph.get_opset_imports();
            auto model = common::make_unique<Model>(std::move(model_proto));
            return Subgraph{std::move(model), parent_graph};
        }

    } // namespace onnx_import

} // namespace ngraph
