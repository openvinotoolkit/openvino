// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "onnx_common/utils.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "place.hpp"
#include "utils/common.hpp"
#include "utils/tensor_external_data.hpp"

using namespace ov::frontend::onnx::common;

namespace ov {
namespace frontend {
namespace onnx {

using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::TensorProto_DataLocation;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ::ONNX_NAMESPACE::TensorProto_DataType_Name;

#define ONNX_INVALID_DATA_TYPE(data_type, expected) \
    OPENVINO_THROW("Invalid data type ", TensorProto_DataType_Name(data_type), " expected: ", expected)
#define ONNX_UNSUPPORTED_DATA_TYPE(data_type, expected) \
    OPENVINO_THROW("Unsupported data type ", TensorProto_DataType_Name(data_type), " expected: ", expected)

namespace detail {
namespace {
template <typename T, typename Container>
inline std::vector<T> __get_data(const Container& container) {
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4267)
#    pragma warning(disable : 4244)
#endif
    return std::vector<T>(std::begin(container), std::end(container));
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}

template <typename T, typename SRC>
inline std::vector<T> __get_data(const void* data, const size_t data_size) {
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4267)
#    pragma warning(disable : 4244)
#endif
    return std::vector<T>(static_cast<const SRC*>(data), static_cast<const SRC*>(data) + data_size);
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}

template <typename T>
inline std::vector<T> __get_raw_data(const std::string& raw_data, int onnx_data_type) {
    auto it = reinterpret_cast<const T*>(raw_data.data());
    return std::vector<T>(it, it + (raw_data.size() / get_onnx_data_size(onnx_data_type)));
}

}  // namespace
}  // namespace detail

using MappedMemoryHandles = std::shared_ptr<std::map<std::string, std::shared_ptr<ov::MappedMemory>>>;
using LocalStreamHandles = std::shared_ptr<std::map<std::string, std::shared_ptr<std::ifstream>>>;

class TensorONNXPlace : public ov::frontend::onnx::TensorPlace {
public:
    TensorONNXPlace(const ov::frontend::InputModel& input_model,
                    const ov::PartialShape& pshape,
                    ov::element::Type type,
                    const std::vector<std::string>& names,
                    const void* data,
                    const size_t data_size,
                    std::shared_ptr<std::string> data_location,
                    const bool is_raw)
        : ov::frontend::onnx::TensorPlace(input_model, pshape, type, names),
          m_input_model(input_model),
          m_data(data),
          m_data_size(data_size),
          m_data_location(data_location),
          m_is_raw(is_raw) {};

    void translate(ov::Output<ov::Node>& output);

    bool is_input() const override {
        return m_input_idx >= 0;
    }
    size_t get_input_index() const {
        FRONT_END_GENERAL_CHECK(is_input(), "This is not input TensorPlace. Can not deliver input index");
        return static_cast<size_t>(m_input_idx);
    }
    bool is_output() const override {
        return m_output_idx >= 0;
    }
    size_t get_output_index() const {
        FRONT_END_GENERAL_CHECK(is_output(), "This is not output TensorPlace. Can not deliver output index");
        return static_cast<size_t>(m_output_idx);
    }
    void set_input_index(const int64_t& idx) {
        m_input_idx = idx;
    }
    void set_output_index(const int64_t& idx) {
        m_output_idx = idx;
    }

    const void* get_data() const {
        return m_data;
    }

    size_t get_data_size() const {
        return m_data_size;
    }

    std::shared_ptr<std::string> get_data_location() const {
        return m_data_location;
    }

    bool is_raw() const {
        return m_is_raw;
    }

    detail::MappedMemoryHandles get_mmap_cache();
    detail::LocalStreamHandles get_stream_cache();

protected:
    int64_t m_input_idx = -1, m_output_idx = -1;
    const ov::frontend::InputModel& m_input_model;
    const void* m_data;
    size_t m_data_size;
    std::shared_ptr<std::string> m_data_location;
    bool m_is_raw;
};

class Tensor {
public:
    enum class Type {
        undefined = TensorProto_DataType::TensorProto_DataType_UNDEFINED,
        float32 = TensorProto_DataType::TensorProto_DataType_FLOAT,
        uint4 = TensorProto_DataType::TensorProto_DataType_UINT4,
        int4 = TensorProto_DataType::TensorProto_DataType_INT4,
        uint8 = TensorProto_DataType::TensorProto_DataType_UINT8,
        int8 = TensorProto_DataType::TensorProto_DataType_INT8,
        uint16 = TensorProto_DataType::TensorProto_DataType_UINT16,
        int16 = TensorProto_DataType::TensorProto_DataType_INT16,
        int32 = TensorProto_DataType::TensorProto_DataType_INT32,
        int64 = TensorProto_DataType::TensorProto_DataType_INT64,
        string = TensorProto_DataType::TensorProto_DataType_STRING,
        boolean = TensorProto_DataType::TensorProto_DataType_BOOL,
        float16 = TensorProto_DataType::TensorProto_DataType_FLOAT16,
        float64 = TensorProto_DataType::TensorProto_DataType_DOUBLE,
        uint32 = TensorProto_DataType::TensorProto_DataType_UINT32,
        uint64 = TensorProto_DataType::TensorProto_DataType_UINT64,
        bfloat16 = TensorProto_DataType::TensorProto_DataType_BFLOAT16,
        complex64 = TensorProto_DataType::TensorProto_DataType_COMPLEX64,
        complex128 = TensorProto_DataType::TensorProto_DataType_COMPLEX128,
        float8e4m3fn = TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN,
        float8e5m2 = TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2,
    };

    Tensor() = delete;
    Tensor(const TensorProto& tensor, const std::string& model_dir, detail::MappedMemoryHandles mmap_cache)
        : m_tensor_proto{&tensor},
          m_tensor_place(nullptr),
          m_shape{std::begin(tensor.dims()), std::end(tensor.dims())},
          m_model_dir{model_dir},
          m_mmap_cache{mmap_cache} {
        if (m_shape == ov::Shape{0}) {
            // It's possible to construct a tensor in ONNX with "dims: 0" property
            // Such tensor contains a scalar. This results in a ov::Shape{0} stored in m_shape.
            // In OpenVINO a scalar is represented with ov::Shape{} and thus this replacement.
            m_shape = ov::Shape{};
        }
    }

    Tensor(const std::shared_ptr<TensorONNXPlace>& tensor_place);

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    const ov::Shape& get_shape() const {
        return m_shape;
    }
    template <typename T>
    std::vector<T> get_data() const {
        if (m_tensor_proto->has_segment()) {
            FRONT_END_THROW("Loading segments isn't supported");
        }
        ONNX_UNSUPPORTED_DATA_TYPE(m_tensor_proto->data_type(), "[nothing expected]");
    }

    const std::string get_name() const {
        if (m_tensor_place != nullptr) {
            const auto& names = m_tensor_place->get_names();
            if (names.size() <= 0)
                FRONT_END_THROW("Tensor has no specified name");
            return names[0];
        }
        if (!m_tensor_proto->has_name()) {
            FRONT_END_THROW("Tensor has no specified name");
        }
        return m_tensor_proto->name();
    }

    Type get_type() const {
        if (m_tensor_place != nullptr) {
            FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
        }
        if (!m_tensor_proto->has_data_type()) {
            FRONT_END_THROW("Tensor has no specified data type");
        }
        return static_cast<Type>(m_tensor_proto->data_type());
    }

    const ov::element::Type& get_ov_type() const {
        if (m_tensor_place != nullptr) {
            return m_tensor_place->get_element_type();
        }
        if (!m_tensor_proto->has_data_type()) {
            FRONT_END_THROW("Tensor has no specified data type");
        }
        switch (m_tensor_proto->data_type()) {
        case TensorProto_DataType::TensorProto_DataType_BOOL:
            return ov::element::boolean;
        case TensorProto_DataType::TensorProto_DataType_FLOAT:
            return ov::element::f32;
        case TensorProto_DataType::TensorProto_DataType_FLOAT16:
            return ov::element::f16;
        case TensorProto_DataType::TensorProto_DataType_DOUBLE:
            return ov::element::f64;
        case TensorProto_DataType::TensorProto_DataType_INT4:
            return ov::element::i4;
        case TensorProto_DataType::TensorProto_DataType_INT8:
            return ov::element::i8;
        case TensorProto_DataType::TensorProto_DataType_INT16:
            return ov::element::i16;
        case TensorProto_DataType::TensorProto_DataType_INT32:
            return ov::element::i32;
        case TensorProto_DataType::TensorProto_DataType_INT64:
            return ov::element::i64;
        case TensorProto_DataType::TensorProto_DataType_UINT4:
            return ov::element::u4;
        case TensorProto_DataType::TensorProto_DataType_UINT8:
            return ov::element::u8;
        case TensorProto_DataType::TensorProto_DataType_UINT16:
            return ov::element::u16;
        case TensorProto_DataType::TensorProto_DataType_UINT32:
            return ov::element::u32;
        case TensorProto_DataType::TensorProto_DataType_UINT64:
            return ov::element::u64;
        case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
            return ov::element::bf16;
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
            return ov::element::f8e4m3;
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
            return ov::element::f8e5m2;
        case TensorProto_DataType::TensorProto_DataType_STRING:
            return ov::element::string;
        case TensorProto_DataType::TensorProto_DataType_UNDEFINED:
            FRONT_END_THROW("Data type is Undefined");
        default:
            ONNX_UNSUPPORTED_DATA_TYPE(
                m_tensor_proto->data_type(),
                "BOOL, BFLOAT16, FLOAT8E4M3FN, FLOAT8E5M2, FLOAT, FLOAT16, DOUBLE, INT4, INT8, INT16, INT32, INT64, "
                "UINT4, UINT8, UINT16, UINT32, UINT64, STRING");
        }
    }

    operator TensorProto_DataType() const {
        return static_cast<TensorProto_DataType>(m_tensor_proto->data_type());
    }

    std::shared_ptr<ov::op::v0::Constant> get_ov_constant() const;

private:
    bool has_external_data() const {
        if (m_tensor_place != nullptr) {
            return m_tensor_place->get_data_location() != nullptr;
        }
        return m_tensor_proto->has_data_location() &&
               m_tensor_proto->data_location() == TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL;
    }

    template <typename T>
    std::vector<T> get_external_data() const {
        const auto ext_data = m_tensor_place != nullptr
                                  ? detail::TensorExternalData(*m_tensor_place->get_data_location(),
                                                               reinterpret_cast<size_t>(m_tensor_place->get_data()),
                                                               m_tensor_place->get_data_size())
                                  : detail::TensorExternalData(*m_tensor_proto);
        std::shared_ptr<ov::AlignedBuffer> buffer = nullptr;
        if (ext_data.data_location() == detail::ORT_MEM_ADDR) {
            buffer = ext_data.load_external_mem_data();
        } else if (m_mmap_cache) {
            buffer = ext_data.load_external_mmap_data(m_model_dir, m_mmap_cache);
        } else {
            buffer = ext_data.load_external_data(m_model_dir);
        }
        return std::vector<T>(buffer->get_ptr<T>(), buffer->get_ptr<T>() + (buffer->size() / sizeof(T)));
    }

    const void* get_data_ptr() const {
        if (has_external_data()) {
            FRONT_END_THROW("Unexpected usage of method for externally stored data");
        }
        if (m_tensor_place != nullptr) {
            return m_tensor_place->get_data();
        }

        if (m_tensor_proto->has_raw_data()) {
            return m_tensor_proto->raw_data().data();
        }
        switch (m_tensor_proto->data_type()) {
        case TensorProto_DataType::TensorProto_DataType_FLOAT:
            return m_tensor_proto->float_data().data();
        case TensorProto_DataType::TensorProto_DataType_INT32:
            return m_tensor_proto->int32_data().data();
        case TensorProto_DataType::TensorProto_DataType_INT64:
            return m_tensor_proto->int64_data().data();
        case TensorProto_DataType::TensorProto_DataType_UINT64:
            return m_tensor_proto->uint64_data().data();
        case TensorProto_DataType::TensorProto_DataType_DOUBLE:
            return m_tensor_proto->double_data().data();
        }
        ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT, INT32, INT64, UINT64, DOUBLE");
    }

    size_t get_data_size() const {
        if (m_tensor_place != nullptr) {
            if (m_tensor_place->is_raw()) {
                return m_tensor_place->get_data_size() /
                       get_onnx_data_size(ov_to_onnx_data_type(m_tensor_place->get_element_type()));
            } else {
                return m_tensor_place->get_data_size();
            }
        }
        if (has_external_data()) {
            const auto ext_data = detail::TensorExternalData(*m_tensor_proto);
            return ext_data.size() / get_onnx_data_size(m_tensor_proto->data_type());
        }
        if (m_tensor_proto->has_raw_data()) {
            return m_tensor_proto->raw_data().size() / get_onnx_data_size(m_tensor_proto->data_type());
        }
        switch (m_tensor_proto->data_type()) {
        case TensorProto_DataType::TensorProto_DataType_FLOAT:
            return m_tensor_proto->float_data_size();
        case TensorProto_DataType::TensorProto_DataType_INT32:
            return m_tensor_proto->int32_data_size();
        case TensorProto_DataType::TensorProto_DataType_INT64:
            return m_tensor_proto->int64_data_size();
        case TensorProto_DataType::TensorProto_DataType_UINT64:
            return m_tensor_proto->uint64_data_size();
        case TensorProto_DataType::TensorProto_DataType_DOUBLE:
            return m_tensor_proto->double_data_size();
        case TensorProto_DataType::TensorProto_DataType_STRING:
            return m_tensor_proto->string_data_size();
        case TensorProto_DataType::TensorProto_DataType_INT4:
        case TensorProto_DataType::TensorProto_DataType_INT8:
        case TensorProto_DataType::TensorProto_DataType_INT16:
        case TensorProto_DataType::TensorProto_DataType_UINT4:
        case TensorProto_DataType::TensorProto_DataType_UINT8:
        case TensorProto_DataType::TensorProto_DataType_UINT16:
        case TensorProto_DataType::TensorProto_DataType_BOOL:
        case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
        case TensorProto_DataType::TensorProto_DataType_FLOAT16:
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
            return m_tensor_proto->int32_data_size();
        }
        ONNX_INVALID_DATA_TYPE(
            m_tensor_proto->data_type(),
            "BOOL, BFLOAT16, FLOAT8E4M3FN, FLOAT8E5M2, FLOAT, FLOAT16, DOUBLE, INT4, INT8, INT16, INT32, INT64, "
            "UINT4, UINT8, UINT16, UINT32, UINT64, STRING");
    }

    const TensorProto* m_tensor_proto;
    std::shared_ptr<TensorONNXPlace> m_tensor_place;
    ov::Shape m_shape;
    std::string m_model_dir;
    detail::MappedMemoryHandles m_mmap_cache;
};

inline std::ostream& operator<<(std::ostream& outs, const Tensor& tensor) {
    return (outs << "<Tensor: " << tensor.get_name() << ">");
}

template <>
std::vector<double> Tensor::get_data() const;

template <>
std::vector<float> Tensor::get_data() const;

template <>
std::vector<ov::float16> Tensor::get_data() const;

template <>
std::vector<ov::bfloat16> Tensor::get_data() const;

template <>
std::vector<int8_t> Tensor::get_data() const;

template <>
std::vector<int16_t> Tensor::get_data() const;

template <>
std::vector<int32_t> Tensor::get_data() const;

template <>
std::vector<int64_t> Tensor::get_data() const;

template <>
std::vector<uint8_t> Tensor::get_data() const;

template <>
std::vector<uint16_t> Tensor::get_data() const;

template <>
std::vector<uint32_t> Tensor::get_data() const;

template <>
std::vector<uint64_t> Tensor::get_data() const;

template <>
std::vector<ov::float8_e4m3> Tensor::get_data() const;

template <>
std::vector<ov::float8_e5m2> Tensor::get_data() const;

template <>
std::vector<char> Tensor::get_data() const;

template <>
std::vector<std::string> Tensor::get_data() const;

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
