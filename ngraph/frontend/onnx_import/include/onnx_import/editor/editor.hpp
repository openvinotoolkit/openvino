//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <map>
#include <memory>

#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_import/utils/onnx_importer_visibility.hpp"

namespace ONNX_NAMESPACE
{
    // forward declaration to avoid the necessity of include paths setting in components
    // that don't directly depend on the ONNX library
    class ModelProto;
} // namespace ONNX_NAMESPACE

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
        class ONNX_IMPORTER_API ONNXModelEditor final
        {
        public:
            ONNXModelEditor() = delete;

            /// \brief Creates an editor from a model file located on a storage device. The file
            ///        is parsed and loaded into the m_model_proto member variable.
            ///
            /// \param model_path Path to the file containing the model.
            ONNXModelEditor(const std::string& model_path);

            /// \brief Modifies the in-memory representation of the model (m_model_proto) by setting
            ///        custom input types for all inputs specified in the provided map.
            ///
            /// \param input_types A collection of pairs {input_name: new_input_type} that should be
            ///                    used to modified the ONNX model loaded from a file. This method
            ///                    throws an exception if the model doesn't contain any of
            ///                    the inputs specified in its parameter.
            void set_input_types(const std::map<std::string, element::Type_t>& input_types);

            /// \brief Modifies the in-memory representation of the model (m_model_proto) by setting
            ///        custom input shapes for all inputs specified in the provided map.
            ///
            /// \param input_shapes A collection of pairs {input_name: new_input_shape} that should
            ///                     be used to modified the ONNX model loaded from a file. This
            ///                     method throws an exception if the model doesn't contain any of
            ///                     the inputs specified in its parameter.
            void set_input_shapes(const std::map<std::string, ngraph::PartialShape>& input_shapes);

            /// \brief Returns a non-const reference to the underlying ModelProto object, possibly
            ///        modified by the editor's API calls
            ///
            /// \return A reference to ONNX ModelProto object containing the in-memory model
            ONNX_NAMESPACE::ModelProto& model() const;

            /// \brief Returns the path to the original model file
            const std::string& model_path() const;

            /// \brief Saves the possibly model held by this class to a file. Serializes in binary
            /// mode.
            ///
            /// \param out_file_path A path to the file where the modified model should be dumped.
            void serialize(const std::string& out_file_path) const;

        private:
            const std::string m_model_path;

            class Impl;
            std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
        };
    } // namespace onnx_import
} // namespace ngraph
