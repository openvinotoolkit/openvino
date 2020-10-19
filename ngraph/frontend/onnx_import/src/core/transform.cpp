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

#include "transform.hpp"

#include "ngraph/file_util.hpp"
#include "onnx_import/ops_bridge.hpp"

void ngraph::onnx_import::transform::fixup_legacy_operators(ONNX_NAMESPACE::GraphProto* graph_proto)
{
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
            auto external_data_full_path =
                file_util::path_join(model_dir_path, external_data_relative_path);

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            file_util::convert_path_win_style(external_data_full_path);
#endif

            // Set full paths to the external file
            initializer_tensor.mutable_external_data(location_key_value_index)
                ->set_value(external_data_full_path);
        }
    }
}