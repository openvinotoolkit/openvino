// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ngraph/function.hpp"
#include "onnx_import/utils/onnx_importer_visibility.hpp"

namespace ONNX_NAMESPACE
{
    class ModelProto;
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            /// \brief      Imports and converts an serialized ONNX model from a ModelProto
            ///             to an nGraph Function representation.
            ///
            /// \note       The function can be used only internally by OV components!
            ///             Passing ModelProto between componets which use different protobuf
            ///             library can cause segfaults. If stream parsing fails or the ONNX model
            ///             contains unsupported ops, the function throws an ngraph_error exception.
            ///
            /// \param[in]  model_proto Reference to a GraphProto object.
            /// \param[in]  model_path  The path to the imported onnx model.
            ///                         It is required if the imported model uses data saved in
            ///                         external files.
            ///
            /// \return     An nGraph function that represents a single output from the created
            /// graph.
            std::shared_ptr<Function> import_onnx_model(ONNX_NAMESPACE::ModelProto& model_proto,
                                                        const std::string& model_path);
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
