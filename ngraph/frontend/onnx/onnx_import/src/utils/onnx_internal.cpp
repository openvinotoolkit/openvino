// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/onnx_pb.h>

#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "core/transform.hpp"
#include "onnx_framework_node.hpp"
#include "onnx_import/core/model.hpp"
#include "onnx_import/utils/onnx_internal.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            void remove_dangling_parameters(std::shared_ptr<Function>& function)
            {
                const auto parameters = function->get_parameters();
                for (auto parameter : parameters)
                {
                    const auto parameter_users = parameter->get_users();
                    // if a Parameter is connected to a ONNXFrameworkNode that was not converted
                    // during convert_function it means, this Parameter is dangling and we can
                    // remove it from function
                    const bool is_dangling_parameter = std::all_of(
                        parameter_users.begin(),
                        parameter_users.end(),
                        [](const std::shared_ptr<ngraph::Node>& node) -> bool {
                            return std::dynamic_pointer_cast<frontend::ONNXFrameworkNode>(node) !=
                                   nullptr;
                        });
                    if (is_dangling_parameter)
                    {
                        function->remove_parameter(parameter);
                    }
                }
            }

            void remove_dangling_results(std::shared_ptr<Function>& function)
            {
                const auto results = function->get_results();
                for (auto result : results)
                {
                    // we can remove Result from function if after function conversion,
                    // Result is connected to NullNode only
                    const auto result_inputs = result->input_values();
                    const bool is_dangling_result =
                        std::all_of(result_inputs.begin(),
                                    result_inputs.end(),
                                    [](const Output<ngraph::Node>& node) -> bool {
                                        return ngraph::op::is_null(node);
                                    });
                    if (is_dangling_result)
                    {
                        function->remove_result(result);
                    }
                }
            }

            void convert_decoded_function(std::shared_ptr<Function> function)
            {
                for (const auto& node : function->get_ordered_ops())
                {
                    if (auto raw_node =
                            std::dynamic_pointer_cast<frontend::ONNXFrameworkNode>(node))
                    {
                        if (auto subgraph_node =
                                std::dynamic_pointer_cast<frontend::ONNXSubgraphFrameworkNode>(
                                    node))
                        {
                            subgraph_node->infer_inputs_from_parent();
                            convert_decoded_function(subgraph_node->get_subgraph_body());
                        }
                        auto ng_nodes = raw_node->get_ng_nodes();
                        replace_node(raw_node, ng_nodes);
                    }
                    else
                    {
                        // Have to revalidate node because new intpus can affect shape/type
                        // propagation for already translated nodes
                        node->revalidate_and_infer_types();
                    }
                }
                detail::remove_dangling_parameters(function);
                detail::remove_dangling_results(function);
            }

            void apply_transformations(ONNX_NAMESPACE::ModelProto& model_proto,
                                       const std::string& model_path)
            {
                transform::expand_onnx_functions(model_proto);
                transform::fixup_legacy_operators(model_proto);
                transform::update_external_data_paths(model_proto, model_path);
            }

            std::shared_ptr<Function>
                import_onnx_model(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                  const std::string& model_path)
            {
                apply_transformations(*model_proto, model_path);
                Graph graph{model_proto};
                return graph.convert();
            }

            std::shared_ptr<Function>
                decode_to_framework_nodes(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                          const std::string& model_path)
            {
                apply_transformations(*model_proto, model_path);
                auto graph = std::make_shared<Graph>(model_proto);
                return graph->decode();
            }
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
