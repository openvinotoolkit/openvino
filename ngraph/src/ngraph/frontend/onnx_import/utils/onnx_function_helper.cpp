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

#include <algorithm>
#include <fstream>
#include <memory>

#include "common.hpp"
#include "default_opset.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx_function_helper.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
        {
            NodeVector expand_onnx_function(const Node& node)
            {
                // Vector of names of operators which return final ouputs
                std::vector<std::string> output_op_names;
                // Vector of nGraph inputs form orginal nGraph node
                std::vector<std::shared_ptr<ngraph::Node>> orginal_inputs = node.get_ng_inputs();
                // Vector of nGraph inputs from expannded nGraph node
                std::vector<std::shared_ptr<ngraph::Node>> helper_inputs;
                // Finla nGraph nodes that should be returned
                NodeVector final_nodes;

                const ONNX_NAMESPACE::NodeProto node_proto = node.node_proto();

                // Create a graph
                ONNX_NAMESPACE::GraphProto graph;

                // Prepare NodeProto with ONNX function
                ONNX_NAMESPACE::NodeProto* new_node = graph.add_node();
                new_node->CopyFrom(node_proto);
                new_node->clear_input();
                new_node->clear_output();

                // Add input to node and graph
                for (auto input : orginal_inputs)
                {
                    // Add input name to NodeProto with ONNX function
                    new_node->add_input(input->get_name());

                    // Add input to graph
                    ONNX_NAMESPACE::ValueInfoProto* proto_input = graph.add_input();
                    proto_input->set_name(input->get_name());
                    auto input_type = input->get_element_type();
                    // Warning: Consider using  PartialShape not just Shape
                    auto input_shape = input->get_output_shape(0);
                    *proto_input->mutable_type() = get_proto_type(input_type, input_shape);
                }

                // Add outputs' names to node and graph
                for (auto output : node.get_output_names())
                {
                    new_node->add_output(output);

                    ONNX_NAMESPACE::ValueInfoProto* y = graph.add_output();
                    y->set_name(output);
                }

                // Get vector of nGraph nodes after expanding ONNX function
                std::vector<std::shared_ptr<ngraph::Node>> nodes =
                    get_nodes_from_onnx_function(new_node, graph, 11);

                // Extract inputs from expanded function and names of operators which return final
                // ouputs
                for (auto node : nodes)
                {
                    if (node->is_parameter())
                    {
                        helper_inputs.push_back(node);
                    }
                    else if (node->is_output())
                    {
                        output_op_names.push_back(node->get_input_node_ptr(0)->get_name());
                    }
                }

                // Swap input in nodes from helping nGrpah function with one from original function
                for (auto node : nodes)
                {
                    for (auto& input : node->inputs())
                    {
                        for (int i = 0; i < helper_inputs.size(); ++i)
                        {
                            if (input.get_source_output() == helper_inputs.at(i))
                            {
                                input.replace_source_output(orginal_inputs.at(i));
                            }
                        }
                    }
                }

                // Obtain final nGraph nodes searching by name
                for (int i = nodes.size() - 1; i >= 0; --i)
                {
                    auto result = std::find(
                        output_op_names.begin(), output_op_names.end(), nodes.at(i)->get_name());

                    if (result != output_op_names.end())
                    {
                        final_nodes.push_back(nodes.at(i));
                    }
                }

                return final_nodes;
            }

            ONNX_NAMESPACE::TypeProto get_proto_type(element::Type type, Shape shape)
            {
                ONNX_NAMESPACE::TypeProto target_type;
                target_type.mutable_tensor_type()->set_elem_type(
                    common::get_proto_element_type(type));

                for (auto dim : shape)
                {
                    target_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
                        dim);
                }
                return target_type;
            }

            std::vector<std::shared_ptr<ngraph::Node>>
                get_nodes_from_onnx_function(ONNX_NAMESPACE::NodeProto* new_node,
                                             ONNX_NAMESPACE::GraphProto graph,
                                             int opset_version)
            {
                const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema(
                    new_node->op_type(), opset_version, "");
                const ONNX_NAMESPACE::FunctionProto* func = schema->GetFunction();

                FunctionExpandHelper(*new_node, *func, graph);

                // Need to erase the node with expanded function since this Pull Request
                // https://github.com/onnx/onnx/pull/2601 is not merged
                graph.mutable_node()->erase(graph.node().begin());

                ONNX_NAMESPACE::ModelProto model;
                auto* graph_ptr = model.mutable_graph();
                *graph_ptr = graph;
                auto* model_opset_version = model.add_opset_import();
                model_opset_version->set_version(11);

                std::istringstream model_stream{model.SerializeAsString()};
                auto function = ngraph::onnx_import::import_onnx_model(model_stream);

                return function->get_ordered_ops();
            }
        }
    }
}
