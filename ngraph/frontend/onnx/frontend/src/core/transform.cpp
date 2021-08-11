// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/defs/function.h>
#include <onnx/defs/schema.h>

#include "core/model.hpp"
#include "core/transform.hpp"

#include "ngraph/file_util.hpp"
#include "ops_bridge.hpp"

void ngraph::onnx_import::transform::expand_onnx_functions(ONNX_NAMESPACE::ModelProto& model_proto)
{
    auto graph_proto = model_proto.mutable_graph();

    for (int i = 0; i < graph_proto->node().size(); ++i)
    {
        ONNX_NAMESPACE::NodeProto node = graph_proto->node().Get(i);

        // Check if node operation is one of the functions we want to expand
        if (std::find(onnx_functions_to_expand.begin(),
                      onnx_functions_to_expand.end(),
                      node.op_type()) == onnx_functions_to_expand.end())
        {
            continue;
        }

        // Retrieve the operation schema from ONNX library
        int opset_version = static_cast<int>(get_opset_version(model_proto, node.domain()));
        const auto* schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
        const auto node_op_schema =
            schema_registry->GetSchema(node.op_type(), opset_version, node.domain());

        // Check if operation schema found
        if (!node_op_schema)
        {
            continue;
        }

        // Check if operation schema contains a function body and expand function
        if (node_op_schema->HasFunction())
        {
            const auto* func_proto = node_op_schema->GetFunction();
            ONNX_NAMESPACE::FunctionExpandHelper(node, *func_proto, *graph_proto);

            // Remove the original node which contained the function.
            graph_proto->mutable_node()->erase(graph_proto->mutable_node()->begin() + i);
        }

        else if (node_op_schema->HasContextDependentFunction())
        {
            ONNX_NAMESPACE::FunctionBodyBuildContextImpl ctx(node);
            ONNX_NAMESPACE::FunctionProto func_proto;
            node_op_schema->BuildContextDependentFunction(ctx, func_proto);
            ONNX_NAMESPACE::FunctionExpandHelper(node, func_proto, *graph_proto);

            // Remove the original node which contained the function.
            graph_proto->mutable_node()->erase(graph_proto->mutable_node()->begin() + i);
        }
    }
}

void ngraph::onnx_import::transform::update_external_data_paths(
    ONNX_NAMESPACE::ModelProto& model_proto, const std::string& model_path)
{
    if (model_path.empty())
    {
        return;
    }
    const auto model_dir_path = file_util::get_directory(model_path);
    auto graph_proto = model_proto.mutable_graph();
    for (auto& initializer_tensor : *graph_proto->mutable_initializer())
    {
        const auto location_key_value_index = 0;
        if (initializer_tensor.has_data_location() &&
            initializer_tensor.data_location() ==
                ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL)
        {
            const auto external_data_relative_path =
                initializer_tensor.external_data(location_key_value_index).value();
            const auto santized_external_data_relative_path =
                file_util::sanitize_path(external_data_relative_path);
            auto external_data_full_path =
                file_util::path_join(model_dir_path, santized_external_data_relative_path);

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            file_util::convert_path_win_style(external_data_full_path);
#endif

            // Set full paths to the external file
            initializer_tensor.mutable_external_data(location_key_value_index)
                ->set_value(external_data_full_path);
        }
    }
}

void ngraph::onnx_import::transform::fixup_legacy_operators(ONNX_NAMESPACE::ModelProto& model_proto)
{
    auto graph_proto = model_proto.mutable_graph();
    for (auto& node : *graph_proto->mutable_node())
    {
        auto it = std::find(legacy_ops_to_fixup.begin(), legacy_ops_to_fixup.end(), node.op_type());
        if (it != legacy_ops_to_fixup.end())
        {
            if (!node.has_domain() || node.domain().empty() || node.domain() == "ai.onnx")
            {
                node.set_domain(OPENVINO_ONNX_DOMAIN);
            }
        }
    }
}
