// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/onnx_pb.h>

#include "ngraph/except.hpp"
#include "onnx_common/utils.hpp"

namespace ngraph
{
    namespace onnx_common
    {
        size_t get_onnx_data_size(int32_t onnx_type)
        {
            switch (onnx_type)
            {
            case ONNX_NAMESPACE::TensorProto_DataType_BOOL: return sizeof(char);
            case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: return 2 * sizeof(double);
            case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: return 2 * sizeof(float);
            case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: return sizeof(double);
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: return 2;
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: return sizeof(float);
            case ONNX_NAMESPACE::TensorProto_DataType_INT8: return sizeof(int8_t);
            case ONNX_NAMESPACE::TensorProto_DataType_INT16: return sizeof(int16_t);
            case ONNX_NAMESPACE::TensorProto_DataType_INT32: return sizeof(int32_t);
            case ONNX_NAMESPACE::TensorProto_DataType_INT64: return sizeof(int64_t);
            case ONNX_NAMESPACE::TensorProto_DataType_UINT8: return sizeof(uint8_t);
            case ONNX_NAMESPACE::TensorProto_DataType_UINT16: return sizeof(uint16_t);
            case ONNX_NAMESPACE::TensorProto_DataType_UINT32: return sizeof(uint32_t);
            case ONNX_NAMESPACE::TensorProto_DataType_UINT64: return sizeof(uint64_t);
            }
#ifdef NGRAPH_USE_PROTOBUF_LITE
            throw ngraph_error("unsupported element type");
#else
            throw ngraph_error("unsupported element type: " +
                               ONNX_NAMESPACE::TensorProto_DataType_Name(
                                   static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_type)));
#endif
        }
    } // namespace onnx_editor
} // namespace ngraph
