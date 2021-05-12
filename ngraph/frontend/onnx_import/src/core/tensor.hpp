// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>
#include <utility>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_common/utils.hpp"
#include "utils/common.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // Detecting automatically the underlying type used to store the information
        // for data type of values a tensor is holding. A bug was discovered in protobuf
        // which forced ONNX team to switch from `enum TensorProto_DataType` to `int32`
        // in order to workaround the bug. This line allows using both versions of ONNX
        // generated wrappers.
        using TensorProto_DataType = decltype(ONNX_NAMESPACE::TensorProto{}.data_type());

        namespace error
        {
            namespace tensor
            {
                struct invalid_data_type : ngraph_error
                {
#ifdef NGRAPH_USE_PROTOBUF_LITE
                    explicit invalid_data_type(TensorProto_DataType type)
                        : ngraph_error{"invalid data type"}
                    {
                    }
#else
                    explicit invalid_data_type(TensorProto_DataType type)
                        : ngraph_error{"invalid data type: " +
                                       ONNX_NAMESPACE::TensorProto_DataType_Name(
                                           static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type))}
                    {
                    }
#endif
                };

                struct unsupported_data_type : ngraph_error
                {
#ifdef NGRAPH_USE_PROTOBUF_LITE
                    explicit unsupported_data_type(TensorProto_DataType type)
                        : ngraph_error{"unsupported data type"}
                    {
                    }
#else
                    explicit unsupported_data_type(TensorProto_DataType type)
                        : ngraph_error{"unsupported data type: " +
                                       ONNX_NAMESPACE::TensorProto_DataType_Name(
                                           static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type))}
                    {
                    }
#endif
                };

                struct unspecified_name : ngraph_error
                {
                    unspecified_name()
                        : ngraph_error{"tensor has no name specified"}
                    {
                    }
                };

                struct unspecified_data_type : ngraph_error
                {
                    unspecified_data_type()
                        : ngraph_error{"tensor has no data type specified"}
                    {
                    }
                };

                struct data_type_undefined : ngraph_error
                {
                    data_type_undefined()
                        : ngraph_error{"data type is not defined"}
                    {
                    }
                };

                struct segments_unsupported : ngraph_error
                {
                    segments_unsupported()
                        : ngraph_error{"loading segments not supported"}
                    {
                    }
                };
            } // namespace tensor
        }     // namespace error

        namespace detail
        {
            namespace tensor
            {
                namespace
                {
                    namespace detail
                    {
                        template <typename T, typename Container>
                        inline std::vector<T> __get_data(const Container& container)
                        {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
                            return std::vector<T>(std::begin(container), std::end(container));
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
                        }

                        template <typename T>
                        inline std::vector<T> __get_raw_data(const std::string& raw_data,
                                                             int onnx_data_type)
                        {
                            auto it = reinterpret_cast<const T*>(raw_data.data());
                            return std::vector<T>(
                                it,
                                it + (raw_data.size() /
                                      onnx_common::get_onnx_data_size(onnx_data_type)));
                        }

                        template <typename T>
                        inline std::vector<T>
                            get_external_data(const ONNX_NAMESPACE::TensorProto& tensor)
                        {
                            const auto tensor_external_data = TensorExternalData(tensor);
                            const auto raw_data = tensor_external_data.load_external_data();

                            return detail::__get_raw_data<T>(raw_data, tensor.data_type());
                        }

                        bool has_tensor_external_data(const ONNX_NAMESPACE::TensorProto& tensor)
                        {
                            if (tensor.has_data_location() &&
                                tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::
                                                              TensorProto_DataLocation_EXTERNAL)
                            {
                                return true;
                            }
                            else
                            {
                                return false;
                            }
                        }
                    } // namespace detail
                }     // namespace

                template <typename T>
                inline std::vector<T> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    throw ngraph::onnx_import::error::tensor::unsupported_data_type{
                        tensor.data_type()};
                }

                template <>
                inline std::vector<double> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<double>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<double>(tensor.raw_data(),
                                                              tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)
                    {
                        return detail::__get_data<double>(tensor.double_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
                    {
                        return detail::__get_data<double>(tensor.float_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32)
                    {
                        return detail::__get_data<double>(tensor.int32_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64)
                    {
                        return detail::__get_data<double>(tensor.int64_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT64)
                    {
                        return detail::__get_data<double>(tensor.uint64_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<float> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<float>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<float>(tensor.raw_data(), tensor.data_type());
                    }
                    if ((tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
                    {
                        return detail::__get_data<float>(tensor.float_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32)
                    {
                        return detail::__get_data<float>(tensor.int32_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64)
                    {
                        return detail::__get_data<float>(tensor.int64_data());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT64)
                    {
                        return detail::__get_data<float>(tensor.uint64_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<ngraph::float16>
                    get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<float16>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<ngraph::float16>(tensor.raw_data(),
                                                                       tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)
                    {
                        return detail::__get_data<ngraph::float16>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<ngraph::bfloat16>
                    get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<bfloat16>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<ngraph::bfloat16>(tensor.raw_data(),
                                                                        tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)
                    {
                        return detail::__get_data<ngraph::bfloat16>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int8_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<int8_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int8_t>(tensor.raw_data(),
                                                              tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8)
                    {
                        return detail::__get_data<int8_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int16_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<int16_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int16_t>(tensor.raw_data(),
                                                               tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT16)
                    {
                        return detail::__get_data<int16_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int32_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<int32_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int32_t>(tensor.raw_data(),
                                                               tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32)
                    {
                        return detail::__get_data<int32_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int64_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<int64_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int64_t>(tensor.raw_data(),
                                                               tensor.data_type());
                    }
                    if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64)
                    {
                        throw error::tensor::invalid_data_type{tensor.data_type()};
                    }
                    return detail::__get_data<int64_t>(tensor.int64_data());
                }

                template <>
                inline std::vector<uint8_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<uint8_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint8_t>(tensor.raw_data(),
                                                               tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8)
                    {
                        return detail::__get_data<uint8_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<uint16_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<uint16_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint16_t>(tensor.raw_data(),
                                                                tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT16)
                    {
                        return detail::__get_data<uint16_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<uint32_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<uint32_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint32_t>(tensor.raw_data(),
                                                                tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT32)
                    {
                        return detail::__get_data<uint32_t>(tensor.uint64_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<uint64_t> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<uint64_t>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint64_t>(tensor.raw_data(),
                                                                tensor.data_type());
                    }
                    if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT64)
                    {
                        throw error::tensor::invalid_data_type{tensor.data_type()};
                    }
                    return detail::__get_data<uint64_t>(tensor.uint64_data());
                }

                template <>
                inline std::vector<char> get_data(const ONNX_NAMESPACE::TensorProto& tensor)
                {
                    // Boolean values are stored as char because std::vector<bool>
                    // can behave differently from other vector containers.
                    if (detail::has_tensor_external_data(tensor))
                    {
                        return detail::get_external_data<char>(tensor);
                    }
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<char>(tensor.raw_data(), tensor.data_type());
                    }
                    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_BOOL)
                    {
                        return detail::__get_data<char>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }
            } // namespace tensor
        }     // namespace detail

        class Tensor
        {
        public:
            enum class Type
            {
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
            explicit Tensor(const ONNX_NAMESPACE::TensorProto& tensor)
                : m_tensor_proto{&tensor}
                , m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}
            {
                if (m_shape == Shape{0})
                {
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

            const Shape& get_shape() const { return m_shape; }
            template <typename T>
            std::vector<T> get_data() const
            {
                if (m_tensor_proto->has_segment())
                {
                    throw error::tensor::segments_unsupported{};
                }
                return detail::tensor::get_data<T>(*m_tensor_proto);
            }

            const std::string& get_name() const
            {
                if (!m_tensor_proto->has_name())
                {
                    throw error::tensor::unspecified_name{};
                }
                return m_tensor_proto->name();
            }

            Type get_type() const
            {
                if (!m_tensor_proto->has_data_type())
                {
                    throw error::tensor::unspecified_data_type{};
                }
                return static_cast<Type>(m_tensor_proto->data_type());
            }

            const element::Type& get_ng_type() const
            {
                if (!m_tensor_proto->has_data_type())
                {
                    throw error::tensor::unspecified_data_type{};
                }
                switch (m_tensor_proto->data_type())
                {
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
                default: throw error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
                }
            }

            operator TensorProto_DataType() const { return m_tensor_proto->data_type(); }
            std::shared_ptr<ngraph::op::Constant> get_ng_constant() const
            {
                switch (m_tensor_proto->data_type())
                {
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
                default: throw error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
                }
            }

        private:
            template <typename T>
            std::shared_ptr<ngraph::op::Constant> make_ng_constant(const element::Type& type) const
            {
                auto constant =
                    std::make_shared<ngraph::op::Constant>(type, m_shape, get_data<T>());
                if (m_tensor_proto->has_name())
                {
                    constant->set_friendly_name(get_name());
                }
                return constant;
            }

            const ONNX_NAMESPACE::TensorProto* m_tensor_proto;
            Shape m_shape;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Tensor& tensor)
        {
            return (outs << "<Tensor: " << tensor.get_name() << ">");
        }
    } // namespace onnx_import
} // namespace ngraph
