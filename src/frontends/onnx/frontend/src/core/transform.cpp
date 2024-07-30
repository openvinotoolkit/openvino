// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log.hpp"
#if defined(_MSC_VER)
#    pragma warning(push)
// Protobuf: conversion from 'XXX' to 'YYY', possible loss of data
#    pragma warning(disable : 4244)
#endif

#include <onnx/defs/function.h>
#include <onnx/defs/schema.h>
#include <onnx/shape_inference/implementation.h>

#include <algorithm>

#include "core/model.hpp"
#include "core/transform.hpp"
#include "openvino/util/log.hpp"
#include "ops_bridge.hpp"

using namespace ::ONNX_NAMESPACE;

namespace ov {
namespace frontend {
namespace onnx {
namespace transform {
namespace {
TypeProto get_input_type(std::string const& name, GraphProto& graph) {
    for (const auto& input : graph.input()) {
        if (input.name() == name) {
            return input.type();
        }
    }
    for (const auto& initializer : graph.initializer()) {
        if (initializer.name() == name) {
            TypeProto ret;
            auto* tensor_type = ret.mutable_tensor_type();
            tensor_type->set_elem_type(initializer.data_type());

            auto* tensor_shape = tensor_type->mutable_shape();
            tensor_shape->clear_dim();
            const auto& initializer_dims = initializer.dims();
            for (auto&& dim : initializer_dims) {
                auto* new_dim = tensor_shape->add_dim();
                new_dim->set_dim_value(dim);
            }
            return ret;
        }
    }
    for (const auto& value_info : graph.value_info()) {
        if (value_info.name() == name) {
            return value_info.type();
        }
    }
    return TypeProto();
}

void function_expand_and_remove_original_node(const NodeProto& node,
                                              const FunctionProto& func_proto,
                                              GraphProto* graph,
                                              int current_node_idx) {
    const auto before_expand_size = graph->node().size();
    FunctionExpandHelper(node, func_proto, *graph);
    const auto added_nodes = graph->node().size() - before_expand_size;

    // Remove the original node which contained the function
    graph->mutable_node()->erase(graph->mutable_node()->begin() + current_node_idx);

    // Move nodes from expanded function to position of removed node
    std::rotate(graph->mutable_node()->begin() + current_node_idx,
                graph->mutable_node()->end() - added_nodes,
                graph->mutable_node()->end());
}

}  // namespace
}  // namespace transform
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

void ov::frontend::onnx::transform::expand_onnx_functions(ModelProto& model_proto) {
    auto graph_proto = model_proto.mutable_graph();

    for (int i = 0; i < graph_proto->node().size(); ++i) {
        NodeProto node = graph_proto->node().Get(i);

        // Check if node operation is one of the functions we want to expand
        if (std::find(onnx_functions_to_expand.begin(), onnx_functions_to_expand.end(), node.op_type()) ==
            onnx_functions_to_expand.end()) {
            continue;
        }

        // Retrieve the operation schema from ONNX library
        int opset_version = static_cast<int>(get_opset_version(model_proto, node.domain()));
        const auto* schema_registry = OpSchemaRegistry::Instance();
        const auto node_op_schema = schema_registry->GetSchema(node.op_type(), opset_version, node.domain());

        // Check if operation schema found
        if (!node_op_schema) {
            continue;
        }

        // Check if operation schema contains a function body and expand function
        if (node_op_schema->HasFunction()) {
            const auto* func_proto = node_op_schema->GetFunction();
            // Move index to the previous position because a first node of expanded function can have also function
            function_expand_and_remove_original_node(node, *func_proto, graph_proto, i--);
        }

        else if (node_op_schema->HasContextDependentFunction()) {
            // In order to expand a context-dependent function, we need to infer types
            try {
                shape_inference::InferShapes(model_proto);
#ifdef ENABLE_OPENVINO_DEBUG
            } catch (const std::exception& e) {
                OPENVINO_WARN("ONNX ov::Shape inference failed: ", e.what());
            }
#else
            } catch (const std::exception&) {
            }
#endif
            std::vector<TypeProto> input_types;
            for (const auto& input : node.input()) {
                input_types.push_back(get_input_type(input, *graph_proto));
            }

            FunctionBodyBuildContextImpl ctx(node, input_types);
            FunctionProto func_proto;
            node_op_schema->BuildContextDependentFunction(ctx, func_proto);
            // Move index to the previous position because a first node of expanded function can have also function
            function_expand_and_remove_original_node(node, func_proto, graph_proto, i--);
        }
    }
}

void ov::frontend::onnx::transform::fixup_legacy_operators(ModelProto& model_proto) {
    auto graph_proto = model_proto.mutable_graph();
    for (auto& node : *graph_proto->mutable_node()) {
        auto it = std::find(legacy_ops_to_fixup.begin(), legacy_ops_to_fixup.end(), node.op_type());
        if (it != legacy_ops_to_fixup.end()) {
            if (!node.has_domain() || node.domain().empty() || node.domain() == "ai.onnx") {
                node.set_domain(OPENVINO_ONNX_DOMAIN);
            }
        }
    }
}

#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
