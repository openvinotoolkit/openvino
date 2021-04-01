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
            ONNX_IMPORTER_API
            std::shared_ptr<Function> import_onnx_model(ONNX_NAMESPACE::ModelProto& model_proto,
                                                        const std::string& model_path);
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
