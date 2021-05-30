// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>
#include <vector>

#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class SparseTensor
        {
        public:
            SparseTensor() = delete;
            explicit SparseTensor(const ONNX_NAMESPACE::SparseTensorProto& sparse_tensor)
                : m_sparse_tensor_proto{&sparse_tensor}
                , m_values{sparse_tensor.values()}
                , m_indices{sparse_tensor.indices()}
                , m_shape{std::begin(sparse_tensor.dims()), std::end(sparse_tensor.dims())}
            {
                if (m_shape == Shape{0})
                {
                    // It's possible to construct a sparse tensor in ONNX with "dims: 0" property
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

        private:
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
