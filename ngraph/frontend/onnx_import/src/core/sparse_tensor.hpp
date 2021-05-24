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
#include "tensor.hpp"
#include "utils/common.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        using TensorProto_DataType = decltype(ONNX_NAMESPACE::TensorProto{}.data_type());

        class SparseTensor
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

            SparseTensor() = delete;
            explicit SparseTensor(const ONNX_NAMESPACE::SparseTensorProto& sparse_tensor)
                : m_sparse_tensor_proto{&sparse_tensor}
                , m_values{sparse_tensor.values()}
                , m_indices{sparse_tensor.indices()}
                , m_shape{std::begin(sparse_tensor.dims()), std::end(sparse_tensor.dims())}
            {
                if (m_shape == Shape{0})
                {
                    // It's possible to construct a tensor in ONNX with "dims: 0" property
                    // Such tensor contains a scalar. This results in a Shape{0} stored in m_shape.
                    // In nGraph a scalar is represented with Shape{} and thus this replacement.
                    m_shape = Shape{};
                }
            }

            SparseTensor(const SparseTensor&) = default;
            SparseTensor(SparseTensor&&) = default;

            SparseTensor& operator=(const SparseTensor&) = delete;
            SparseTensor& operator=(SparseTensor&&) = delete;

            const Shape& get_shape() const { return m_shape; }

            const std::string& get_name() const { return m_values.get_name(); }

            const Tensor& get_values() const { return m_values; }

            const Tensor& get_indices() const { return m_indices; }

            const element::Type& get_ng_type() const { return m_values.get_ng_type(); }

            std::shared_ptr<ngraph::op::Constant> get_ng_constant() const
            {
                switch (TensorProto_DataType(m_values))
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
                default: throw error::tensor::unsupported_data_type{TensorProto_DataType(m_values)};
                }
            }

        private:
            template <typename T>
            std::shared_ptr<ngraph::op::Constant> make_ng_constant(const element::Type& type) const
            {
                auto constant =
                    std::make_shared<ngraph::op::Constant>(type, m_shape, std::vector<T>{});
                return constant;
            }
            const ONNX_NAMESPACE::SparseTensorProto* m_sparse_tensor_proto;
            Tensor m_values;
            Tensor m_indices;
            Shape m_shape;
        };

        inline std::ostream& operator<<(std::ostream& outs, const SparseTensor& tensor)
        {
            return (outs << "<Sparse Tensor>");
        }
    } // namespace onnx_import
} // namespace ngraph
