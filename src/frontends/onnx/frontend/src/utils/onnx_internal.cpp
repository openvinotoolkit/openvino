// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/onnx_internal.hpp"

#include <onnx/onnx_pb.h>

#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "core/transform.hpp"
#include "onnx_framework_node.hpp"
#include "openvino/core/model.hpp"
#include "openvino/util/file_util.hpp"

using namespace ov;

namespace ov::frontend::onnx::detail {
namespace {
void remove_dangling_parameters(std::shared_ptr<ov::Model>& model) {
    const auto parameters = model->get_parameters();
    for (auto parameter : parameters) {
        const auto parameter_users = parameter->get_users();
        // if a Parameter is connected to a ONNXFrameworkNode that was not converted
        // during convert_function it means, this Parameter is dangling and we can
        // remove it from function
        const bool is_dangling_parameter =
            std::all_of(parameter_users.begin(),
                        parameter_users.end(),
                        [](const std::shared_ptr<ov::Node>& node) -> bool {
                            return ov::as_type_ptr<ov::frontend::onnx::ONNXFrameworkNode>(node) != nullptr;
                        });
        if (is_dangling_parameter) {
            model->remove_parameter(parameter);
        }
    }
}

void remove_dangling_results(std::shared_ptr<ov::Model>& model) {
    const auto results = model->get_results();
    for (auto result : results) {
        // we can remove Result from function if after function conversion,
        // Result is connected to NullNode only
        const auto result_inputs = result->input_values();
        const bool is_dangling_result =
            std::all_of(result_inputs.begin(), result_inputs.end(), [](const Output<ov::Node>& node) -> bool {
                return ov::op::util::is_null(node);
            });
        if (is_dangling_result) {
            model->remove_result(result);
        }
    }
}

void apply_transformations(ModelProto& model_proto) {
    transform::fixup_legacy_operators(model_proto);
}

}  // namespace

void convert_decoded_model(std::shared_ptr<ov::Model> model) {
    auto& rt_info = model->get_rt_info();
    auto it = rt_info.find(ONNX_GRAPH_RT_ATTRIBUTE);
    OPENVINO_ASSERT(it != rt_info.end(),
                    "Could not find '" + std::string(ONNX_GRAPH_RT_ATTRIBUTE) +
                        "' attribute in decoded model. Model probably wasn't created by FrontEnd::decode function.");
    auto onnx_graph = it->second.as<std::shared_ptr<ov::frontend::onnx::Graph>>();
    for (const auto& node : model->get_ordered_ops()) {
        if (auto raw_node = ov::as_type_ptr<ov::frontend::onnx::ONNXFrameworkNode>(node)) {
            if (auto subgraph_node = ov::as_type_ptr<ov::frontend::onnx::ONNXSubgraphFrameworkNode>(node)) {
                subgraph_node->infer_inputs_from_parent();
                for (auto& model : subgraph_node->get_subgraph_models()) {
                    convert_decoded_model(model);
                }
            }
            auto ov_nodes = raw_node->get_ov_nodes(onnx_graph);
            replace_node(raw_node, ov_nodes);
        } else {
            // Have to revalidate node because new intpus can affect shape/type
            // propagation for already translated nodes
            node->revalidate_and_infer_types();
        }
    }
    rt_info.erase(it);
    detail::remove_dangling_parameters(model);
    detail::remove_dangling_results(model);
}

std::shared_ptr<ov::Model> import_onnx_model(std::shared_ptr<ModelProto> model_proto,
                                             const std::string& model_path,
                                             detail::MappedMemoryHandles mmap_cache,
                                             ov::frontend::ExtensionHolder extensions) {
    apply_transformations(*model_proto);
    Graph graph{ov::util::get_directory(model_path).string(), model_proto, mmap_cache, std::move(extensions)};
    return graph.convert();
}

std::shared_ptr<ov::Model> decode_to_framework_nodes(std::shared_ptr<ModelProto> model_proto,
                                                     const std::string& model_path,
                                                     detail::MappedMemoryHandles mmap_cache,
                                                     ov::frontend::ExtensionHolder extensions) {
    apply_transformations(*model_proto);
    auto graph =
        std::make_shared<Graph>(ov::util::get_directory(model_path).string(), model_proto, mmap_cache, extensions);
    return graph->decode();
}
}  // namespace ov::frontend::onnx::detail
