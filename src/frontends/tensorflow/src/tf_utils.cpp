// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tf_utils.hpp"

#include <stdint.h>

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;

namespace {

template <typename T>
void extract_tensor_content(const std::string& tensor_content, ov::Tensor* values) {
    const auto tensor_content_size = tensor_content.size();
    FRONT_END_GENERAL_CHECK(tensor_content_size % sizeof(T) == 0,
                            "Size of tensor_content (",
                            tensor_content_size,
                            ") is not a multiple of ",
                            sizeof(T));

    const T* tensor_values = reinterpret_cast<const T*>(tensor_content.data());
    FRONT_END_GENERAL_CHECK(values->get_size() == tensor_content_size / sizeof(T),
                            "Size of tensor is not equal to tensor_content size.");
    std::copy(tensor_values, tensor_values + tensor_content_size / sizeof(T), values->data<T>());
}

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)  // possible loss of data
#    pragma warning(disable : 4267)  // possible loss of data
#endif
template <typename T>
void extract_compressed_tensor_content(const ::tensorflow::TensorProto& tensor_proto,
                                       int64_t val_size,
                                       ov::Tensor* values) {
    auto val_lastsaved = static_cast<T>(0);
    auto values_data = values->data<T>();
    for (size_t i = 0; i < values->get_size(); i++) {
        if (val_size == 0) {
            values_data[i] = static_cast<T>(0);
        } else if (static_cast<int64_t>(i) < val_size) {
            auto val_i = static_cast<T>(0);
            switch (values->get_element_type()) {
            // TODO: there are more element types to support here
            case ov::element::boolean:
                val_i = tensor_proto.bool_val()[i];
                break;
            case ov::element::i32:
                val_i = tensor_proto.int_val()[i];
                break;
            case ov::element::i64:
                val_i = tensor_proto.int64_val()[i];
                break;
            case ov::element::f16:
                val_i = float16::from_bits(tensor_proto.half_val()[i]);
                break;
            case ov::element::f32:
                val_i = tensor_proto.float_val()[i];
                break;
            case ov::element::f64:
                val_i = tensor_proto.double_val()[i];
                break;
            default:
                FRONT_END_THROW("Encountered unknown element type " + values->get_element_type().get_type_name());
            }
            values_data[i] = val_i;
            val_lastsaved = val_i;
        } else {
            values_data[i] = val_lastsaved;
        }
    }
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace

ov::element::Type ov::frontend::tensorflow::get_ov_type(const ::tensorflow::DataType& type) {
    static const std::map<::tensorflow::DataType, ov::element::Type> type_map{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16}};

    auto it = type_map.find(type);
    // for all unsupported types return dynamic type
    return it == type_map.end() ? ov::element::dynamic : it->second;
}

ov::Any ov::frontend::tensorflow::unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto) {
    return unpack_tensor_proto(tensor_proto, tensor_proto.tensor_shape(), tensor_proto.dtype());
}

ov::Any ov::frontend::tensorflow::unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto,
                                                      const ::tensorflow::TensorShapeProto& tensor_shape,
                                                      const ::tensorflow::DataType& tensor_type) {
    ov::PartialShape pshape;
    for (int i = 0; i < tensor_shape.dim_size(); i++) {
        pshape.push_back(tensor_shape.dim(i).size());
    }
    FRONT_END_GENERAL_CHECK(pshape.is_static(), "Dynamic shapes are not supported for Tensor attribute.");
    ov::element::Type ov_type = get_ov_type(tensor_type);

    if (tensor_type != ::tensorflow::DataType::DT_STRING) {
        FRONT_END_GENERAL_CHECK(
            ov_type.is_static(),
            "Encountered unknown element type " + DataType_Name(tensor_type) + " on an empty tensor_proto");
    } else {
        auto data = std::vector<std::string>();
        for (const auto& item : tensor_proto.string_val()) {
            data.push_back(item);
        }
        return data;
    }
    ov::Tensor res(ov_type, pshape.get_shape());
    auto tensor_content = tensor_proto.tensor_content();
    if (!tensor_content.empty() && tensor_proto.has_tensor_shape()) {
        switch (ov_type) {
        case ov::element::u8:
            extract_tensor_content<uint8_t>(tensor_content, &res);
            break;
        case ov::element::i8:
            extract_tensor_content<int8_t>(tensor_content, &res);
            break;
        case ov::element::i16:
            extract_tensor_content<int16_t>(tensor_content, &res);
            break;
        case ov::element::i32:
            extract_tensor_content<int32_t>(tensor_content, &res);
            break;
        case ov::element::i64:
            extract_tensor_content<int64_t>(tensor_content, &res);
            break;
        case ov::element::f16:
            extract_tensor_content<float16>(tensor_content, &res);
            break;
        case ov::element::f32:
            extract_tensor_content<float>(tensor_content, &res);
            break;
        case ov::element::f64:
            extract_tensor_content<double>(tensor_content, &res);
            break;
        case ov::element::bf16:
            extract_tensor_content<bfloat16>(tensor_content, &res);
            break;
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    } else {
        int64_t val_size = 0;
        switch (ov_type) {
        case ov::element::boolean:
            val_size = tensor_proto.bool_val_size();
            extract_compressed_tensor_content<bool>(tensor_proto, val_size, &res);
            break;
        case ov::element::i32:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<int32_t>(tensor_proto, val_size, &res);
            break;
        case ov::element::i64:
            val_size = tensor_proto.int64_val_size();
            extract_compressed_tensor_content<int64_t>(tensor_proto, val_size, &res);
            break;
        case ov::element::f16:
            val_size = tensor_proto.half_val_size();
            extract_compressed_tensor_content<float16>(tensor_proto, val_size, &res);
            break;
        case ov::element::f32:
            val_size = tensor_proto.float_val_size();
            extract_compressed_tensor_content<float>(tensor_proto, val_size, &res);
            break;
        case ov::element::f64:
            val_size = tensor_proto.double_val_size();
            extract_compressed_tensor_content<double>(tensor_proto, val_size, &res);
            break;
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    }
    return res;
}
