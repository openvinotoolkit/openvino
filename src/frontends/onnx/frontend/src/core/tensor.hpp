// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_common/utils.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "utils/common.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph {
namespace onnx_import {
// Detecting automatically the underlying type used to store the information
// for data type of values a tensor is holding. A bug was discovered in protobuf
// which forced ONNX team to switch from `enum TensorProto_DataType` to `int32`
// in order to workaround the bug. This line allows using both versions of ONNX
// generated wrappers.
using TensorProto_DataType = decltype(ONNX_NAMESPACE::TensorProto{}.data_type());

namespace error {
namespace tensor {
OPENVINO_SUPPRESS_DEPRECATED_START
struct invalid_data_type : ngraph_error {
    explicit invalid_data_type(TensorProto_DataType type) : ngraph_error{"invalid data type"} {}
};

struct unsupported_data_type : ngraph_error {
    explicit unsupported_data_type(TensorProto_DataType type) : ngraph_error{"unsupported data type"} {}
};

struct unspecified_name : ngraph_error {
    unspecified_name() : ngraph_error{"tensor has no name specified"} {}
};

struct unspecified_data_type : ngraph_error {
    unspecified_data_type() : ngraph_error{"tensor has no data type specified"} {}
};

struct data_type_undefined : ngraph_error {
    data_type_undefined() : ngraph_error{"data type is not defined"} {}
};

struct segments_unsupported : ngraph_error {
    segments_unsupported() : ngraph_error{"loading segments not supported"} {}
};

struct shape_doesnt_match_data_size : ngraph_error {
    shape_doesnt_match_data_size() : ngraph_error{"tensor shape doesn't match data size"} {}
};
OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace tensor
}  // namespace error

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

template <typename T>
inline std::vector<T> __get_raw_data(const std::string& raw_data, int onnx_data_type) {
    auto it = reinterpret_cast<const T*>(raw_data.data());
    return std::vector<T>(it, it + (raw_data.size() / onnx_common::get_onnx_data_size(onnx_data_type)));
}

}  // namespace
}  // namespace detail

class Tensor {
public:
    enum class Type {
        undefined = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
        float32 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
        uint8 = ONNX_NAMESPACE::TensorProto_DataType_UINT8,
        int8 = ONNX_NAMESPACE::TensorProto_DataType_INT8,
        uint16 = ONNX_NAMESPACE::TensorProto_DataType_UINT16,
        int16 = ONNX_NAMESPACE::TensorProto_DataType_INT16,
        int32 = ONNX_NAMESPACE::TensorProto_DataType_INT32,
        int64 = ONNX_NAMESPACE::TensorProto_DataType_INT64,
        string = ONNX_NAMESPACE::TensorProto_DataType_STRING,
        boolean = ONNX_NAMESPACE::TensorProto_DataType_BOOL,
        float16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
        float64 = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
        uint32 = ONNX_NAMESPACE::TensorProto_DataType_UINT32,
        uint64 = ONNX_NAMESPACE::TensorProto_DataType_UINT64,
        bfloat16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
        complex64 = ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64,
        complex128 = ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128
    };

    Tensor() = delete;
    Tensor(const ONNX_NAMESPACE::TensorProto& tensor,
           const std::string& model_dir,
           detail::MappedMemoryHandles mmap_cache)
        : m_tensor_proto{&tensor},
          m_shape{std::begin(tensor.dims()), std::end(tensor.dims())},
          m_model_dir{model_dir},
          m_mmap_cache{mmap_cache} {
        if (m_shape == Shape{0}) {
            // It's possible to construct a tensor in ONNX with "dims: 0" property
            // Such tensor contains a scalar. This results in a Shape{0} stored in m_shape.
            // In nGraph a scalar is represented with Shape{} and thus this replacement.
            m_shape = Shape{};
        }
    }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    const Shape& get_shape() const {
        return m_shape;
    }
    template <typename T>
    std::vector<T> get_data() const {
        if (m_tensor_proto->has_segment()) {
            throw error::tensor::segments_unsupported{};
        }
        throw ngraph::onnx_import::error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
    }

    const std::string& get_name() const {
        if (!m_tensor_proto->has_name()) {
            throw error::tensor::unspecified_name{};
        }
        return m_tensor_proto->name();
    }

    Type get_type() const {
        if (!m_tensor_proto->has_data_type()) {
            throw error::tensor::unspecified_data_type{};
        }
        return static_cast<Type>(m_tensor_proto->data_type());
    }

    const element::Type& get_ng_type() const {
        if (!m_tensor_proto->has_data_type()) {
            throw error::tensor::unspecified_data_type{};
        }
        switch (m_tensor_proto->data_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
            return element::boolean;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
            return element::f32;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
            return element::f16;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
            return element::f64;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
            return element::i8;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
            return element::i16;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
            return element::i32;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
            return element::i64;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
            return element::u8;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
            return element::u16;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
            return element::u32;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
            return element::u64;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16:
            return element::bf16;
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
            throw error::tensor::data_type_undefined{};
        default:
            throw error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
        }
    }

    operator TensorProto_DataType() const {
        return m_tensor_proto->data_type();
    }

    std::shared_ptr<ngraph::op::Constant> get_ng_constant() const {
        if (m_tensor_proto->has_segment()) {
            throw error::tensor::segments_unsupported{};
        }
        switch (m_tensor_proto->data_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
            return make_ng_constant<char>(element::boolean);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
            return make_ng_constant<float>(element::f32);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
            return make_ng_constant<ngraph::float16>(element::f16);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
            return make_ng_constant<double>(element::f64);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
            return make_ng_constant<int8_t>(element::i8);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
            return make_ng_constant<int16_t>(element::i16);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
            return make_ng_constant<int32_t>(element::i32);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
            return make_ng_constant<int64_t>(element::i64);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
            return make_ng_constant<uint8_t>(element::u8);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
            return make_ng_constant<uint16_t>(element::u16);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
            return make_ng_constant<uint32_t>(element::u32);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
            return make_ng_constant<uint64_t>(element::u64);
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16:
            return make_ng_constant<ngraph::bfloat16>(element::bf16);
        default:
            throw error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
        }
    }

private:
    template <typename T,
              typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value ||
                                          std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
                                          std::is_same<T, uint64_t>::value,
                                      bool>::type = true>
    std::shared_ptr<ngraph::op::Constant> make_ng_constant(const element::Type& type) const {
        std::shared_ptr<default_opset::Constant> constant{nullptr};
        size_t data_size = get_data_size();
        if (has_external_data()) {
            const auto ext_data = detail::TensorExternalData(*m_tensor_proto);
            if (m_mmap_cache) {
                constant =
                    std::make_shared<ngraph::op::Constant>(type,
                                                           m_shape,
                                                           ext_data.load_external_mmap_data(m_model_dir, m_mmap_cache));
            } else {
                constant =
                    std::make_shared<ngraph::op::Constant>(type, m_shape, ext_data.load_external_data(m_model_dir));
            }
            if (constant->get_byte_size() != ov::shape_size(m_shape) * type.size()) {
                throw error::invalid_external_data(
                    "The size of the external data file does not match the byte size of an initializer '" + get_name() +
                    "' in the model");
            }
        } else if (data_size == shape_size(m_shape)) {
            constant = std::make_shared<ngraph::op::Constant>(type, m_shape, get_data_ptr());
        } else if (data_size == 0 && m_shape.size() == 0) {
            constant = common::make_failsafe_constant(type);
        } else {
            throw error::tensor::shape_doesnt_match_data_size{};
        }

        if (m_tensor_proto->has_name()) {
            constant->set_friendly_name(get_name());
        }
        return constant;
    }

    template <typename T,
              typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, double>::value &&
                                          !std::is_same<T, int32_t>::value && !std::is_same<T, int64_t>::value &&
                                          !std::is_same<T, uint64_t>::value,
                                      bool>::type = true>
    std::shared_ptr<ngraph::op::Constant> make_ng_constant(const element::Type& type) const {
        std::shared_ptr<default_opset::Constant> constant{nullptr};
        auto data = get_data<T>();
        auto data_size = data.size();
        if (data_size == shape_size(m_shape)) {
            constant = std::make_shared<ngraph::op::Constant>(type, m_shape, data);
        } else if (data_size == 0 && m_shape.size() == 0) {
            constant = common::make_failsafe_constant(type);
        } else {
            throw error::tensor::shape_doesnt_match_data_size{};
        }
        if (m_tensor_proto->has_name()) {
            constant->set_friendly_name(get_name());
        }
        return constant;
    }

    bool has_external_data() const {
        return m_tensor_proto->has_data_location() &&
               m_tensor_proto->data_location() ==
                   ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL;
    }

    template <typename T>
    std::vector<T> get_external_data() const {
        const auto ext_data = detail::TensorExternalData(*m_tensor_proto);
        std::shared_ptr<ov::AlignedBuffer> buffer = nullptr;
        if (m_mmap_cache) {
            buffer = ext_data.load_external_mmap_data(m_model_dir, m_mmap_cache);
        } else {
            buffer = ext_data.load_external_data(m_model_dir);
        }
        return std::vector<T>(buffer->get_ptr<char>(), buffer->get_ptr<char>() + buffer->size());
    }

    const void* get_data_ptr() const {
        if (m_tensor_proto->has_raw_data()) {
            return m_tensor_proto->raw_data().data();
        }
        switch (m_tensor_proto->data_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            return m_tensor_proto->float_data().data();
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
            return m_tensor_proto->int32_data().data();
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
            return m_tensor_proto->int64_data().data();
        case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
            return m_tensor_proto->uint64_data().data();
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
            return m_tensor_proto->double_data().data();
        }
        throw error::tensor::invalid_data_type{m_tensor_proto->data_type()};
    }

    size_t get_data_size() const {
        if (m_tensor_proto->has_raw_data()) {
            return m_tensor_proto->raw_data().size() / onnx_common::get_onnx_data_size(m_tensor_proto->data_type());
        }
        switch (m_tensor_proto->data_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            return m_tensor_proto->float_data_size();
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
            return m_tensor_proto->int32_data_size();
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
            return m_tensor_proto->int64_data_size();
        case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
            return m_tensor_proto->uint64_data_size();
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
            return m_tensor_proto->double_data_size();
        }
        throw error::tensor::invalid_data_type{m_tensor_proto->data_type()};
    }

    const ONNX_NAMESPACE::TensorProto* m_tensor_proto;
    Shape m_shape;
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
std::vector<ngraph::float16> Tensor::get_data() const;

template <>
std::vector<ngraph::bfloat16> Tensor::get_data() const;

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
std::vector<char> Tensor::get_data() const;

}  // namespace onnx_import
}  // namespace ngraph
