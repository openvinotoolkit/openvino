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

#include <istream>
#include <memory>
#include "onnx_import/utils/onnx_importer_visibility.hpp"

namespace ONNX_NAMESPACE
{
    // forward declaration to avoid the necessity of include paths setting in components
    // that don't directly depend on the ONNX library
    class ModelProto;
}

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief A class representing a set of utilities allowing modification of an ONNX model
        ///
        /// \note This class can be used to modify an ONNX model before it gets translated to
        ///       an ngraph::Function by the import_onnx_model function. It lets you modify the
        ///       model's input types and shapes, extract a subgraph and more. An instance of this
        ///       class can be passed directly to the onnx_importer API.
        class ONNX_IMPORTER_API ONNXModelEditor
        {
        public:
            ONNXModelEditor() = delete;
            ~ONNXModelEditor() = default;

            ONNXModelEditor(std::istream& model_stream);

        private:
            std::unique_ptr<ONNX_NAMESPACE::ModelProto> m_model_proto;
        };
    } // namespace onnx_import
} // namespace ngraph
