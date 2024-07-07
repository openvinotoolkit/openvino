// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <vector>

#include "core/attribute.hpp"
#include "core/operator_set.hpp"
#include "core/sparse_tensor.hpp"
#include "core/tensor.hpp"
#include "openvino/frontend/exception.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {
template <typename T>
std::vector<T> get_dense_vector(const std::vector<T>& values, const std::vector<int64_t>& indices, const size_t size) {
    FRONT_END_GENERAL_CHECK(values.size() == indices.size(),
                            "The number of values and indices is not equal."
                            " Indices number: ",
                            indices.size(),
                            " Values number: ",
                            values.size());

    std::vector<T> dense_values(size);
    for (size_t i = 0; i < values.size(); ++i) {
        dense_values.at(indices.at(i)) = values.at(i);
    }
    return dense_values;
}

template <typename T>
std::shared_ptr<v0::Constant> make_dense_tensor_as_constant(const std::vector<int64_t>& indices,
                                                            const Tensor& values_tensor,
                                                            const ov::Shape& shape) {
    auto values = values_tensor.get_data<T>();
    auto dense_vector = get_dense_vector<T>(values, indices, shape_size(shape));
    return v0::Constant::create(values_tensor.get_ov_type(), shape, dense_vector);
}

std::shared_ptr<v0::Constant> get_dense_tensor_as_constant(const std::vector<int64_t>& absolute_indices,
                                                           const Tensor& values_tensor,
                                                           const ov::Shape& shape) {
    switch (values_tensor.get_ov_type()) {
    case ov::element::boolean:
        return make_dense_tensor_as_constant<char>(absolute_indices, values_tensor, shape);
    case ov::element::f32:
        return make_dense_tensor_as_constant<float>(absolute_indices, values_tensor, shape);
    case ov::element::f16:
        return make_dense_tensor_as_constant<ov::float16>(absolute_indices, values_tensor, shape);
    case ov::element::f64:
        return make_dense_tensor_as_constant<double>(absolute_indices, values_tensor, shape);
    case ov::element::i8:
        return make_dense_tensor_as_constant<int8_t>(absolute_indices, values_tensor, shape);
    case ov::element::i16:
        return make_dense_tensor_as_constant<int16_t>(absolute_indices, values_tensor, shape);
    case ov::element::i32:
        return make_dense_tensor_as_constant<int32_t>(absolute_indices, values_tensor, shape);
    case ov::element::i64:
        return make_dense_tensor_as_constant<int64_t>(absolute_indices, values_tensor, shape);
    case ov::element::u8:
        return make_dense_tensor_as_constant<uint8_t>(absolute_indices, values_tensor, shape);
    case ov::element::u16:
        return make_dense_tensor_as_constant<uint16_t>(absolute_indices, values_tensor, shape);
    case ov::element::u32:
        return make_dense_tensor_as_constant<uint32_t>(absolute_indices, values_tensor, shape);
    case ov::element::u64:
        return make_dense_tensor_as_constant<uint64_t>(absolute_indices, values_tensor, shape);
    case ov::element::bf16:
        return make_dense_tensor_as_constant<ov::bfloat16>(absolute_indices, values_tensor, shape);
    default:
        FRONT_END_THROW("Tensor has an unsupported data type");
    }
}

std::vector<int64_t> get_absolute_indices(const Tensor& indices_tensor, const ov::Shape& shape, const size_t& nnz) {
    auto rank = shape.size();
    auto indices = indices_tensor.get_data<int64_t>();
    auto indices_shape = indices_tensor.get_shape();
    std::vector<int64_t> absolute_indices{};
    for (size_t i = 0; i < nnz; ++i) {
        int64_t index = 0;
        for (size_t j = 0; j < rank; ++j) {
            auto dim_index_in_indices = i * rank + j;
            auto dim_value_in_indices = indices.at(dim_index_in_indices);

            if (j < rank - 1) {
                size_t elements_num_per_shape = 1;
                for (size_t k = j + 1; k < rank; ++k)
                    elements_num_per_shape *= shape.at(k);
                index += dim_value_in_indices * elements_num_per_shape;
            } else {
                index += dim_value_in_indices;
            }
        }
        absolute_indices.push_back(index);
    }
    return absolute_indices;
}
}  // namespace

namespace opset_1 {
ov::OutputVector constant(const ov::frontend::onnx::Node& node) {
    auto tensor = node.get_attribute_value<Tensor>("value");
    return {tensor.get_ov_constant()};
}

ONNX_OP("Constant", OPSET_RANGE(1, 12), ai_onnx::opset_1::constant);
}  // namespace opset_1

namespace opset_13 {
ov::OutputVector constant(const ov::frontend::onnx::Node& node) {
    auto attributes_names = node.get_attribute_names();
    FRONT_END_GENERAL_CHECK(attributes_names.size() == 1,
                            "The Constant op expects exactly one attribute."
                            "Got: ",
                            attributes_names.size());

    auto& attribute = node.get_attribute(attributes_names[0]);

    if (attribute.is_float()) {
        return {v0::Constant::create(ov::element::f32, ov::Shape{}, {attribute.get_float()})};
    } else if (attribute.is_float_array()) {
        auto values = attribute.get_float_array();
        return {v0::Constant::create(ov::element::f32, ov::Shape{values.size()}, values)};
    } else if (attribute.is_integer()) {
        return {v0::Constant::create(ov::element::i64, ov::Shape{}, {attribute.get_integer()})};
    } else if (attribute.is_integer_array()) {
        auto values = attribute.get_integer_array();
        return {v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values)};
    } else if (attribute.is_sparse_tensor()) {
        auto sparse_tensor = attribute.get_sparse_tensor();
        const Tensor& values_tensor = sparse_tensor.get_values();
        const Tensor& indices_tensor = sparse_tensor.get_indices();
        const ov::Shape& shape = sparse_tensor.get_shape();
        auto rank = shape.size();
        // NNZ - the number of non-zero values in the sparse-tensor
        auto nnz = values_tensor.get_shape().at(0);
        std::vector<int64_t> absolute_indices{};

        // Check if indices tensor with rank 2 has correct shape [NNZ, rank].
        // [i,j]-th value corresponds to the j-th index of the i-th value (in the
        // values tensor)
        if (indices_tensor.get_shape().size() == 2) {
            FRONT_END_GENERAL_CHECK(indices_tensor.get_shape().at(0) == nnz,
                                    "The number of values and indices is not equal."
                                    " Indices number: ",
                                    indices_tensor.get_shape().at(0),
                                    " Values number: ",
                                    nnz);

            FRONT_END_GENERAL_CHECK(indices_tensor.get_shape().at(1) == rank,
                                    "The indices are incorrect. The second dimension of "
                                    "indices is not equal to the rank of output."
                                    " Second dimension of indices: ",
                                    indices_tensor.get_shape().at(0),
                                    " Rank of output: ",
                                    rank);

            absolute_indices = get_absolute_indices(indices_tensor, shape, nnz);
        }
        // Check if indices tensor with rank 1 has correct shape [NNZ].
        // i-th value is the linearized-index of the i-th value (in the values
        // tensor)
        else {
            FRONT_END_GENERAL_CHECK(indices_tensor.get_shape().at(0) == nnz,
                                    "The number of values and indices is not equal."
                                    " Indices number: ",
                                    indices_tensor.get_shape().at(0),
                                    " Values number: ",
                                    nnz);

            absolute_indices = indices_tensor.get_data<int64_t>();
        }
        return {get_dense_tensor_as_constant(absolute_indices, values_tensor, shape)};
    }
    auto tensor = node.get_attribute_value<Tensor>(attributes_names[0]);
    return {tensor.get_ov_constant()};
}
ONNX_OP("Constant", OPSET_SINCE(13), ai_onnx::opset_13::constant);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
