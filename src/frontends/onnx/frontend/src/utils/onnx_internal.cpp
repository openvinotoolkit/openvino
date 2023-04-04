// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/onnx_internal.hpp"

#include <onnx/onnx_pb.h>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/transform.hpp"
#include "ngraph/file_util.hpp"
#include "onnx_framework_node.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/util/file_util.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {
namespace {
void remove_dangling_parameters(std::shared_ptr<Function>& function) {
    const auto parameters = function->get_parameters();
    for (auto parameter : parameters) {
        const auto parameter_users = parameter->get_users();
        // if a Parameter is connected to a ONNXFrameworkNode that was not converted
        // during convert_function it means, this Parameter is dangling and we can
        // remove it from function
        const bool is_dangling_parameter =
            std::all_of(parameter_users.begin(),
                        parameter_users.end(),
                        [](const std::shared_ptr<ngraph::Node>& node) -> bool {
                            return std::dynamic_pointer_cast<frontend::ONNXFrameworkNode>(node) != nullptr;
                        });
        if (is_dangling_parameter) {
            function->remove_parameter(parameter);
        }
    }
}

void remove_dangling_results(std::shared_ptr<Function>& function) {
    const auto results = function->get_results();
    for (auto result : results) {
        // we can remove Result from function if after function conversion,
        // Result is connected to NullNode only
        const auto result_inputs = result->input_values();
        const bool is_dangling_result =
            std::all_of(result_inputs.begin(), result_inputs.end(), [](const Output<ngraph::Node>& node) -> bool {
                return ngraph::op::is_null(node);
            });
        if (is_dangling_result) {
            function->remove_result(result);
        }
    }
}

void apply_transformations(ONNX_NAMESPACE::ModelProto& model_proto) {
    transform::fixup_legacy_operators(model_proto);
}

}  // namespace

void convert_decoded_function(std::shared_ptr<Function> function) {
    auto& rt_info = function->get_rt_info();
    auto it = rt_info.find(ONNX_GRAPH_RT_ATTRIBUTE);
    OPENVINO_ASSERT(it != rt_info.end(),
                    "Could not find '" + std::string(ONNX_GRAPH_RT_ATTRIBUTE) +
                        "' attribute in decoded model. Model probably wasn't created by FrontEnd::decode function.");
    auto onnx_graph = it->second.as<std::shared_ptr<onnx_import::Graph>>();
    for (const auto& node : function->get_ordered_ops()) {
        if (auto raw_node = std::dynamic_pointer_cast<frontend::ONNXFrameworkNode>(node)) {
            if (auto subgraph_node = std::dynamic_pointer_cast<frontend::ONNXSubgraphFrameworkNode>(node)) {
                subgraph_node->infer_inputs_from_parent();
                for (auto& function : subgraph_node->get_subgraph_functions()) {
                    convert_decoded_function(function);
                }
            }
            auto ng_nodes = raw_node->get_ng_nodes(onnx_graph);
            replace_node(raw_node, ng_nodes);
        } else {
            // Have to revalidate node because new intpus can affect shape/type
            // propagation for already translated nodes
            node->revalidate_and_infer_types();
        }
    }
    rt_info.erase(it);
    detail::remove_dangling_parameters(function);
    detail::remove_dangling_results(function);
}

std::shared_ptr<Function> import_onnx_model(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                            const std::string& model_path,
                                            ov::frontend::ExtensionHolder extensions) {
    apply_transformations(*model_proto);
    NGRAPH_SUPPRESS_DEPRECATED_START
    Graph graph{file_util::get_directory(ov::util::get_absolute_file_path(model_path)),
                model_proto,
                std::move(extensions)};
    NGRAPH_SUPPRESS_DEPRECATED_END
    return graph.convert();
}

std::shared_ptr<Function> decode_to_framework_nodes(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                                    const std::string& model_path,
                                                    ov::frontend::ExtensionHolder extensions) {
    apply_transformations(*model_proto);
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto graph = std::make_shared<Graph>(file_util::get_directory(ov::util::get_absolute_file_path(model_path)),
                                         model_proto,
                                         extensions);
    NGRAPH_SUPPRESS_DEPRECATED_END
    return graph->decode();
}
}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
