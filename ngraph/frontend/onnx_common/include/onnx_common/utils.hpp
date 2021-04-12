// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph/type/element_type.hpp"

namespace ONNX_NAMESPACE
{
    enum TensorProto_DataType;
}

namespace ngraph
{
    namespace onnx_common
    {
        size_t get_onnx_data_size(int32_t onnx_type);
        element::Type_t onnx_to_ng_data_type(const ONNX_NAMESPACE::TensorProto_DataType& onnx_type);
        ONNX_NAMESPACE::TensorProto_DataType ng_to_onnx_data_type(const element::Type_t& ng_type);
        bool is_supported_ng_type(const element::Type_t& ng_type);

    } // namespace onnx_editor
} // namespace ngraph
