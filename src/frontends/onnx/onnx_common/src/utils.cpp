// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_common/utils.hpp"

#include "openvino/core/except.hpp"

using namespace ::ONNX_NAMESPACE;

namespace ov {
namespace frontend {
namespace onnx {
namespace common {
size_t get_onnx_data_size(int32_t onnx_type) {
    switch (onnx_type) {
    case TensorProto_DataType_BOOL:
        return sizeof(char);
    case TensorProto_DataType_COMPLEX128:
        return 2 * sizeof(double);
    case TensorProto_DataType_COMPLEX64:
        return 2 * sizeof(float);
    case TensorProto_DataType_DOUBLE:
        return sizeof(double);
    case TensorProto_DataType_FLOAT16:
        return 2;
    case TensorProto_DataType_FLOAT:
        return sizeof(float);
    case TensorProto_DataType_FLOAT8E4M3FN:
        return sizeof(ov::float8_e4m3);
    case TensorProto_DataType_FLOAT8E5M2:
        return sizeof(ov::float8_e5m2);
    case TensorProto_DataType_INT4:
        return sizeof(int8_t);
    case TensorProto_DataType_INT8:
        return sizeof(int8_t);
    case TensorProto_DataType_INT16:
        return sizeof(int16_t);
    case TensorProto_DataType_INT32:
        return sizeof(int32_t);
    case TensorProto_DataType_INT64:
        return sizeof(int64_t);
    case TensorProto_DataType_UINT4:
        return sizeof(uint8_t);
    case TensorProto_DataType_UINT8:
        return sizeof(uint8_t);
    case TensorProto_DataType_UINT16:
        return sizeof(uint16_t);
    case TensorProto_DataType_UINT32:
        return sizeof(uint32_t);
    case TensorProto_DataType_UINT64:
        return sizeof(uint64_t);
    case TensorProto_DataType_BFLOAT16:
        return sizeof(uint16_t);
    }
    OPENVINO_THROW("unsupported element type");
}
const std::map<ov::element::Type_t, TensorProto_DataType> OV_2_ONNX_TYPES = {
    {ov::element::Type_t::bf16, TensorProto_DataType::TensorProto_DataType_BFLOAT16},
    {ov::element::Type_t::f8e4m3, TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN},
    {ov::element::Type_t::f8e5m2, TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2},
    {ov::element::Type_t::f16, TensorProto_DataType::TensorProto_DataType_FLOAT16},
    {ov::element::Type_t::f32, TensorProto_DataType::TensorProto_DataType_FLOAT},
    {ov::element::Type_t::f64, TensorProto_DataType::TensorProto_DataType_DOUBLE},
    {ov::element::Type_t::i4, TensorProto_DataType::TensorProto_DataType_INT4},
    {ov::element::Type_t::i8, TensorProto_DataType::TensorProto_DataType_INT8},
    {ov::element::Type_t::i16, TensorProto_DataType::TensorProto_DataType_INT16},
    {ov::element::Type_t::i32, TensorProto_DataType::TensorProto_DataType_INT32},
    {ov::element::Type_t::i64, TensorProto_DataType::TensorProto_DataType_INT64},
    {ov::element::Type_t::u4, TensorProto_DataType::TensorProto_DataType_UINT4},
    {ov::element::Type_t::u8, TensorProto_DataType::TensorProto_DataType_UINT8},
    {ov::element::Type_t::u16, TensorProto_DataType::TensorProto_DataType_UINT16},
    {ov::element::Type_t::u32, TensorProto_DataType::TensorProto_DataType_UINT32},
    {ov::element::Type_t::u64, TensorProto_DataType::TensorProto_DataType_UINT64},
    {ov::element::Type_t::boolean, TensorProto_DataType::TensorProto_DataType_BOOL},
    {ov::element::Type_t::string, TensorProto_DataType::TensorProto_DataType_STRING}};

ov::element::Type_t onnx_to_ov_data_type(const TensorProto_DataType& onnx_type) {
    const auto result = std::find_if(OV_2_ONNX_TYPES.begin(),
                                     OV_2_ONNX_TYPES.end(),
                                     [&onnx_type](const std::pair<ov::element::Type_t, TensorProto_DataType>& pair) {
                                         return pair.second == onnx_type;
                                     });
    if (result == std::end(OV_2_ONNX_TYPES)) {
        OPENVINO_THROW("unsupported element type: " +
                       TensorProto_DataType_Name(static_cast<TensorProto_DataType>(onnx_type)));
    }
    return result->first;
}

TensorProto_DataType ov_to_onnx_data_type(const ov::element::Type_t& ov_type) {
    return OV_2_ONNX_TYPES.at(ov_type);
}

bool is_supported_ov_type(const ov::element::Type_t& ov_type) {
    return OV_2_ONNX_TYPES.count(ov_type) > 0;
}

PartialShape onnx_to_ov_shape(const TensorShapeProto& onnx_shape) {
    if (onnx_shape.dim_size() == 0) {
        return Shape{};  // empty list of dimensions denotes a scalar
    }

    std::vector<ov::Dimension> dims;
    for (const auto& onnx_dim : onnx_shape.dim()) {
        if (onnx_dim.has_dim_value()) {
            dims.emplace_back(onnx_dim.dim_value());
        } else  // has_dim_param() == true or it is empty dim
        {
            dims.push_back(ov::Dimension::dynamic());
        }
    }
    return PartialShape{dims};
}

}  // namespace common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
