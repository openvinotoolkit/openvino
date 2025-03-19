// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "tensor.hpp"

namespace ov {
namespace frontend {
namespace onnx {
using ::ONNX_NAMESPACE::SparseTensorProto;

class SparseTensor {
public:
    SparseTensor() = delete;
    SparseTensor(const SparseTensorProto& sparse_tensor,
                 const std::string& model_dir,
                 detail::MappedMemoryHandles mmap_cache)
        : m_values{sparse_tensor.values(), model_dir, mmap_cache},
          m_indices{sparse_tensor.indices(), model_dir, mmap_cache},
          m_shape{std::begin(sparse_tensor.dims()), std::end(sparse_tensor.dims())} {
        if (m_shape == ov::Shape{0}) {
            // It's possible to construct a sparse tensor in ONNX with "dims: 0" property
            // Such tensor contains a scalar. This results in a ov::Shape{0} stored in m_shape.
            // In OpenVINO a scalar is represented with ov::Shape{} and thus this replacement.
            m_shape = ov::Shape{};
        }
    }

    SparseTensor(const SparseTensor&) = default;
    SparseTensor(SparseTensor&&) = default;

    SparseTensor& operator=(const SparseTensor&) = delete;
    SparseTensor& operator=(SparseTensor&&) = delete;

    const ov::Shape& get_shape() const {
        return m_shape;
    }

    const std::string& get_name() const {
        return m_values.get_name();
    }

    const Tensor& get_values() const {
        return m_values;
    }

    const Tensor& get_indices() const {
        return m_indices;
    }

    const ov::element::Type& get_ov_type() const {
        return m_values.get_ov_type();
    }

private:
    Tensor m_values;
    Tensor m_indices;
    ov::Shape m_shape;
};

inline std::ostream& operator<<(std::ostream& outs, const SparseTensor& tensor) {
    return (outs << "<Sparse Tensor>");
}
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
