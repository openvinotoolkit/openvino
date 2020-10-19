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

#pragma once

#include <onnx/onnx_pb.h>

namespace ngraph
{
    namespace onnx_import
    {
        namespace transform
        {
            static const std::vector<std::string> legacy_ops_to_fixup = {
                "DetectionOutput", "FakeQuantize", "GroupNorm", "Normalize", "PriorBox"};

            /// \brief Add support for models with custom operators mistakenly registered in
            ///        "ai.onnx" domain.
            ///
            /// Some legacy models use custom operators (listed in legacy_ops_to_fixup vector) which
            /// were registered in the default ONNX domain. This function updates nodes with these
            /// operations to use OPENVINO_ONNX_DOMAIN in order to process them correctly
            /// in the nGraph ONNX Importer.
            ///
            /// \param graph_proto Protobuf message with graph to transform.
            void fixup_legacy_operators(ONNX_NAMESPACE::GraphProto* graph_proto);

            /// \brief Replace external_data path in tensors with full path to data file.
            ///
            /// Paths to external data files are stored as relative to model path.
            /// This transformation replaces them with a full filesystem path.
            /// As a result in further processing data from external files can be read directly.
            ///
            /// \param graph_proto Protobuf message with graph to transform.
            /// \param model_path Filesystem path to the ONNX model file.
            void update_external_data_paths(ONNX_NAMESPACE::ModelProto& model_proto,
                                            const std::string& model_path);

        } // namespace transform

    } // namespace onnx_import

} // namespace ngraph
