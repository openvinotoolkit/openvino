// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/tensor.hpp"

namespace ov {
namespace frontend {
namespace onnx {

template <>
std::vector<double> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<double>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<double>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
        return detail::__get_data<double>(m_tensor_proto->double_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "DOUBLE, raw data");
}

template <>
std::vector<float> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<float>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<float>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT) {
        return detail::__get_data<float>(m_tensor_proto->float_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT, raw data");
}

template <>
std::vector<ov::float16> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::float16>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::float16>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
        using std::begin;
        using std::end;

        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::float16> float16_data;
        float16_data.reserve(int32_data.size());
        std::transform(begin(int32_data), end(int32_data), std::back_inserter(float16_data), [](int32_t elem) {
            return ov::float16::from_bits(static_cast<uint16_t>(elem));
        });

        return detail::__get_data<ov::float16>(float16_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT16, raw data");
}

template <>
std::vector<ov::bfloat16> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::bfloat16>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::bfloat16>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
        return detail::__get_data<ov::bfloat16>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT32, raw data");
}

template <>
std::vector<int8_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int8_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int8_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT8 ||
        m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT4) {
        return detail::__get_data<int8_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT4, INT8, raw data");
}

template <>
std::vector<int16_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int16_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int16_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT16) {
        return detail::__get_data<int16_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT16, raw data");
}

template <>
std::vector<int32_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int32_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int32_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT32) {
        return detail::__get_data<int32_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT32, raw data");
}

template <>
std::vector<int64_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int64_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int64_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT64) {
        return detail::__get_data<int64_t>(m_tensor_proto->int64_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT64, raw data");
}

template <>
std::vector<uint8_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint8_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint8_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT8 ||
        m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT4) {
        return detail::__get_data<uint8_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT4, UINT8, raw data");
}

template <>
std::vector<uint16_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint16_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint16_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT16) {
        return detail::__get_data<uint16_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT16, raw data");
}

template <>
std::vector<uint32_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint32_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint32_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT32) {
        return detail::__get_data<uint32_t>(m_tensor_proto->uint64_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT32, raw data");
}

template <>
std::vector<uint64_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint64_t>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint64_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT64) {
        return detail::__get_data<uint64_t>(m_tensor_proto->uint64_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT64, raw data");
}

template <>
std::vector<ov::float8_e4m3> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::float8_e4m3>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::float8_e4m3>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN) {
        using std::begin;
        using std::end;

        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::float8_e4m3> float8_data;
        float8_data.reserve(int32_data.size());
        std::transform(begin(int32_data), end(int32_data), std::back_inserter(float8_data), [](int32_t elem) {
            return ov::float8_e4m3::from_bits(static_cast<uint8_t>(elem));
        });

        return detail::__get_data<ov::float8_e4m3>(float8_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT8E4M3, raw data");
}

template <>
std::vector<ov::float8_e5m2> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::float8_e5m2>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::float8_e5m2>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2) {
        using std::begin;
        using std::end;

        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::float8_e5m2> float8_data;
        float8_data.reserve(int32_data.size());
        std::transform(begin(int32_data), end(int32_data), std::back_inserter(float8_data), [](int32_t elem) {
            return ov::float8_e5m2::from_bits(static_cast<uint8_t>(elem));
        });

        return detail::__get_data<ov::float8_e5m2>(float8_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT8E5M2, raw data");
}

template <>
std::vector<char> Tensor::get_data() const {
    // Boolean values are stored as char because std::vector<bool>
    // can behave differently from other vector containers.
    if (has_external_data()) {
        return get_external_data<char>();
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<char>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_BOOL) {
        return detail::__get_data<char>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "BOOL, raw data");
}

template <>
std::vector<std::string> Tensor::get_data() const {
    if (has_external_data()) {
        FRONT_END_THROW("External strings are not supported");
    }
    if (m_tensor_proto->has_raw_data()) {
        FRONT_END_THROW("Loading strings from raw data isn't supported");
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_STRING) {
        return detail::__get_data<std::string>(m_tensor_proto->string_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "STRING");
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
