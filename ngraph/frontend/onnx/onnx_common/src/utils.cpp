// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
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
            case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: return sizeof(uint16_t);
            }
#ifdef NGRAPH_USE_PROTOBUF_LITE
            throw ngraph_error("unsupported element type");
#else
            throw ngraph_error("unsupported element type: " +
                               ONNX_NAMESPACE::TensorProto_DataType_Name(
                                   static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_type)));
#endif
        }
        namespace
        {
            using namespace ONNX_NAMESPACE;
            const std::map<element::Type_t, TensorProto_DataType> NG_2_ONNX_TYPES = {
                {element::Type_t::bf16, TensorProto_DataType::TensorProto_DataType_BFLOAT16},
                {element::Type_t::f16, TensorProto_DataType::TensorProto_DataType_FLOAT16},
                {element::Type_t::f32, TensorProto_DataType::TensorProto_DataType_FLOAT},
                {element::Type_t::f64, TensorProto_DataType::TensorProto_DataType_DOUBLE},
                {element::Type_t::i8, TensorProto_DataType::TensorProto_DataType_INT8},
                {element::Type_t::i16, TensorProto_DataType::TensorProto_DataType_INT16},
                {element::Type_t::i32, TensorProto_DataType::TensorProto_DataType_INT32},
                {element::Type_t::i64, TensorProto_DataType::TensorProto_DataType_INT64},
                {element::Type_t::u8, TensorProto_DataType::TensorProto_DataType_UINT8},
                {element::Type_t::u16, TensorProto_DataType::TensorProto_DataType_UINT16},
                {element::Type_t::u32, TensorProto_DataType::TensorProto_DataType_UINT32},
                {element::Type_t::u64, TensorProto_DataType::TensorProto_DataType_UINT64},
                {element::Type_t::boolean, TensorProto_DataType::TensorProto_DataType_BOOL}};
        } // namespace

        element::Type_t onnx_to_ng_data_type(const TensorProto_DataType& onnx_type)
        {
            const auto result = std::find_if(
                NG_2_ONNX_TYPES.begin(),
                NG_2_ONNX_TYPES.end(),
                [&onnx_type](
                    const std::pair<element::Type_t, ONNX_NAMESPACE::TensorProto_DataType>& pair) {
                    return pair.second == onnx_type;
                });
            if (result == std::end(NG_2_ONNX_TYPES))
            {
                throw ngraph_error(
                    "unsupported element type: " +
                    ONNX_NAMESPACE::TensorProto_DataType_Name(
                        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_type)));
            }
            return result->first;
        }

        TensorProto_DataType ng_to_onnx_data_type(const element::Type_t& ng_type)
        {
            return NG_2_ONNX_TYPES.at(ng_type);
        }

        bool is_supported_ng_type(const element::Type_t& ng_type)
        {
            return NG_2_ONNX_TYPES.count(ng_type) > 0;
        }

    } // namespace onnx_common
} // namespace ngraph
