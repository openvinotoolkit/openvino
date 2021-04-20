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
            const std::map<std::size_t, element::Type_t>& subgraph_inputs_types_map) const
        {
            if (m_attribute_proto->type() != ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH)
            {
                throw error::attribute::InvalidData{m_attribute_proto->type()};
            }

            auto model_proto = common::make_unique<ONNX_NAMESPACE::ModelProto>();

            const auto& graph = m_attribute_proto->g();
            model_proto->mutable_graph()->CopyFrom(graph);

            const std::size_t subgraph_inputs_count =
                static_cast<size_t>(model_proto->mutable_graph()->mutable_input()->size());
            // Use the `subgraph_inputs_types_map` to infer the types for the subgraph inputs
            for (const auto& input_index_type_map : subgraph_inputs_types_map)
            {
                if (input_index_type_map.first >= subgraph_inputs_count)
                {
                    NGRAPH_WARN << "Input with index: '" << input_index_type_map.first
                                << "' was not found in the subgraph";
                }
                else
                {
                    auto subgraph_in =
                        model_proto->mutable_graph()->mutable_input(input_index_type_map.first);
                    subgraph_in->mutable_type()->mutable_tensor_type()->set_elem_type(
                        onnx_common::ng_to_onnx_data_type(input_index_type_map.second));
                }
            }

            // set opset version and domain from the parent graph
            *model_proto->mutable_opset_import() = parent_graph.get_opset_imports();
            auto model = common::make_unique<Model>(std::move(model_proto));
            return Subgraph{std::move(model), parent_graph};
        }

    } // namespace onnx_import

} // namespace ngraph
