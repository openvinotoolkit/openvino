// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/constant.hpp"
#include <vector>
#include "core/attribute.hpp"
#include "core/sparse_tensor.hpp"
#include "core/tensor.hpp"
#include "default_opset.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"


namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                template <typename T>
                inline std::shared_ptr<default_opset::Constant>
                    __make_ng_constant(const element::Type& type, const Tensor& tensor)
                {
                    std::shared_ptr<default_opset::Constant> constant{nullptr};
                    try
                    {
                        constant = std::make_shared<default_opset::Constant>(
                            type, tensor.get_shape(), tensor.get_data<T>());
                    }
                    catch (const ngraph::ngraph_error& exc)
                    {
                        NGRAPH_WARN
                            << "\nCould not create an nGraph Constant for an ONNX Constant "
                               "node. "
                            << "Constant with a 0 value was created instead.\n"
                            << "Verify if the ONNX Constant node contains a correct number of "
                               "elements matching the node's shape. \n"
                            << "Detailed error:\n"
                            << exc.what();
                        constant = std::make_shared<default_opset::Constant>(type, Shape{}, 0);
                    }

                    return constant;
                }

                template <Tensor::Type>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant(const Tensor& tensor)
                {
                    throw error::tensor::unsupported_data_type{tensor};
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::float16>(const Tensor& tensor)
                {
                    return __make_ng_constant<ngraph::float16>(element::f16, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::float32>(const Tensor& tensor)
                {
                    return __make_ng_constant<float>(element::f32, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::float64>(const Tensor& tensor)
                {
                    return __make_ng_constant<double>(element::f64, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::int8>(const Tensor& tensor)
                {
                    return __make_ng_constant<int8_t>(element::i8, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::int16>(const Tensor& tensor)
                {
                    return __make_ng_constant<int16_t>(element::i16, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::int32>(const Tensor& tensor)
                {
                    return __make_ng_constant<int32_t>(element::i32, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::int64>(const Tensor& tensor)
                {
                    return __make_ng_constant<int64_t>(element::i64, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::uint8>(const Tensor& tensor)
                {
                    return __make_ng_constant<uint8_t>(element::u8, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::uint16>(const Tensor& tensor)
                {
                    return __make_ng_constant<uint16_t>(element::u16, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::uint32>(const Tensor& tensor)
                {
                    return __make_ng_constant<uint32_t>(element::u32, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::uint64>(const Tensor& tensor)
                {
                    return __make_ng_constant<uint64_t>(element::u64, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::boolean>(const Tensor& tensor)
                {
                    return __make_ng_constant<char>(element::boolean, tensor);
                }

                template <>
                inline std::shared_ptr<default_opset::Constant>
                    make_ng_constant<Tensor::Type::bfloat16>(const Tensor& tensor)
                {
                    return __make_ng_constant<ngraph::bfloat16>(element::bf16, tensor);
                }

                inline std::shared_ptr<default_opset::Constant> make_constant(const Tensor& tensor)
                {
#define MAKE_NG_CONSTANT(data_type_)                                                               \
    case data_type_: return make_ng_constant<data_type_>(tensor)

                    switch (tensor.get_type())
                    {
                        MAKE_NG_CONSTANT(Tensor::Type::float16);
                        MAKE_NG_CONSTANT(Tensor::Type::float32);
                        MAKE_NG_CONSTANT(Tensor::Type::float64);
                        MAKE_NG_CONSTANT(Tensor::Type::int8);
                        MAKE_NG_CONSTANT(Tensor::Type::int16);
                        MAKE_NG_CONSTANT(Tensor::Type::int32);
                        MAKE_NG_CONSTANT(Tensor::Type::int64);
                        MAKE_NG_CONSTANT(Tensor::Type::uint8);
                        MAKE_NG_CONSTANT(Tensor::Type::uint16);
                        MAKE_NG_CONSTANT(Tensor::Type::uint32);
                        MAKE_NG_CONSTANT(Tensor::Type::uint64);
                        MAKE_NG_CONSTANT(Tensor::Type::boolean);
                        MAKE_NG_CONSTANT(Tensor::Type::bfloat16);
                    default: throw error::tensor::invalid_data_type{tensor};
                    }
                }

                template <typename T>
                std::vector<T> get_dense_vector(const std::vector<T>& values,
                                                const std::vector<int64_t>& indices,
                                                const size_t size)
                {
                    std::vector<T> dense_values(size);
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        dense_values.at(indices.at(i)) = values.at(i);
                    }
                    return dense_values;
                }

                std::shared_ptr<default_opset::Constant>
                    get_dense_tensor_as_constant(const std::vector<int64_t>& absolute_indices,
                                                 const Tensor& values_tensor,
                                                 const Shape& shape)
                {
                    size_t all_elements_number = 1;
                    for (auto dim : shape)
                        all_elements_number *= dim;

                    switch (values_tensor.get_ng_type())
                    {
                    case element::boolean:
                    {
                        auto values = values_tensor.get_data<char>();
                        auto dense_vector =
                            get_dense_vector<char>(values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::f32:
                    {
                        auto values = values_tensor.get_data<float>();
                        auto dense_vector =
                            get_dense_vector<float>(values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::f16:
                    {
                        auto values = values_tensor.get_data<ngraph::float16>();
                        auto dense_vector = get_dense_vector<ngraph::float16>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::f64:
                    {
                        auto values = values_tensor.get_data<double>();
                        auto dense_vector =
                            get_dense_vector<double>(values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::i8:
                    {
                        auto values = values_tensor.get_data<int8_t>();
                        auto dense_vector =
                            get_dense_vector<int8_t>(values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::i16:
                    {
                        auto values = values_tensor.get_data<int16_t>();
                        auto dense_vector = get_dense_vector<int16_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::i32:
                    {
                        auto values = values_tensor.get_data<int32_t>();
                        auto dense_vector = get_dense_vector<int32_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::i64:
                    {
                        auto values = values_tensor.get_data<int64_t>();
                        auto dense_vector = get_dense_vector<int64_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::u8:
                    {
                        auto values = values_tensor.get_data<uint8_t>();
                        auto dense_vector = get_dense_vector<uint8_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::u16:
                    {
                        auto values = values_tensor.get_data<uint16_t>();
                        auto dense_vector = get_dense_vector<uint16_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::u32:
                    {
                        auto values = values_tensor.get_data<uint32_t>();
                        auto dense_vector = get_dense_vector<uint32_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::u64:
                    {
                        auto values = values_tensor.get_data<uint64_t>();
                        auto dense_vector = get_dense_vector<uint64_t>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    case element::bf16:
                    {
                        auto values = values_tensor.get_data<ngraph::bfloat16>();
                        auto dense_vector = get_dense_vector<ngraph::bfloat16>(
                            values, absolute_indices, all_elements_number);
                        return default_opset::Constant::create(
                            values_tensor.get_ng_type(), shape, dense_vector);
                    }
                    default: throw error::tensor::invalid_data_type{values_tensor};
                    }
                }

                std::vector<int64_t> get_absolute_indices(const Tensor& indices_tensor,
                                                          const Shape& shape,
                                                          const size_t& nnz)
                {
                    auto rank = shape.size();
                    auto indices = indices_tensor.get_data<int64_t>();
                    auto indices_shape = indices_tensor.get_shape();
                    std::vector<int64_t> absolute_indices{};
                    for (size_t i = 0; i < nnz; ++i)
                    {
                        int64_t index = 0;
                        for (size_t j = 0; j < rank; ++j)
                        {
                            auto dim_index_in_indices = i * rank + j;
                            auto dim_value_in_indices = indices.at(dim_index_in_indices);

                            if (j < rank - 1)
                            {
                                size_t elements_num_per_shape = 1;
                                for (size_t k = j + 1; k < rank; ++k)
                                    elements_num_per_shape *= shape.at(k);
                                index += dim_value_in_indices * elements_num_per_shape;
                            }
                            else
                            {
                                index += dim_value_in_indices;
                            }
                        }
                        absolute_indices.push_back(index);
                    }
                    return absolute_indices;
                }
            } // namespace

            namespace set_1
            {
                OutputVector constant(const onnx_import::Node& node)
                {
                    return {make_constant(node.get_attribute_value<Tensor>("value"))};
                }

            } // namespace set_1

            namespace set_13
            {
                OutputVector constant(const onnx_import::Node& node)
                {
                    auto attributes_names = node.get_attribute_names();
                    NGRAPH_CHECK(attributes_names.size() == 1,
                                 "The Constant op expects exactly one attribute."
                                 "Got: ",
                                 attributes_names.size());

                    auto& attribute = node.get_attribute(attributes_names[0]);

                    if (attribute.is_float())
                    {
                        return {default_opset::Constant::create(
                            element::f32, ngraph::Shape{}, {attribute.get_float()})};
                    }
                    else if (attribute.is_float_array())
                    {
                        auto values = attribute.get_float_array();
                        return {default_opset::Constant::create(
                            element::f32, ngraph::Shape{values.size()}, values)};
                    }
                    else if (attribute.is_integer())
                    {
                        return {default_opset::Constant::create(
                            element::i64, ngraph::Shape{}, {attribute.get_integer()})};
                    }
                    else if (attribute.is_integer_array())
                    {
                        auto values = attribute.get_integer_array();
                        return {default_opset::Constant::create(
                            element::i64, ngraph::Shape{values.size()}, values)};
                    }
                    else if (attribute.is_sparse_tensor())
                    {
                        auto sparse_tensor = attribute.get_sparse_tensor();
                        const Tensor& values_tensor = sparse_tensor.get_values();
                        const Tensor& indices_tensor = sparse_tensor.get_indices();
                        const Shape& shape = sparse_tensor.get_shape();
                        auto rank = shape.size();
                        // NNZ - the number of non-zero values in the sparse-tensor
                        auto nnz = values_tensor.get_shape().at(0);
                        std::vector<int64_t> absolute_indices{};

                        // Check if indices tensor with rank 2 has correct shape [NNZ, rank].
                        // [i,j]-th value corresponds to the j-th index of the i-th value (in the values tensor)
                        if(indices_tensor.get_shape().size() == 2){
                            NGRAPH_CHECK(indices_tensor.get_shape().at(0) == nnz,
                                    "The number of values and indices is not equal."
                                    " Indices number: ",
                                    indices_tensor.get_shape().at(0),
                                    " Values number: ",
                                    nnz);

                            NGRAPH_CHECK(indices_tensor.get_shape().at(1) == rank,
                                    "The indices are incorrect. The second dimension of indices is not equal to the rank of output."
                                    " Second dimension of indices: ",
                                    indices_tensor.get_shape().at(0),
                                    " Rank of output: ",
                                    rank);

                            absolute_indices = get_absolute_indices(indices_tensor, shape, nnz);
                        }
                        // Check if indices tensor with rank 1 has correct shape [NNZ].
                        // i-th value is the linearized-index of the i-th value (in the values tensor)
                        else
                        {
                            NGRAPH_CHECK(indices_tensor.get_shape().at(0) == nnz,
                                    "The number of values and indices is not equal."
                                    " Indices number: ",
                                    indices_tensor.get_shape().at(0),
                                    " Values number: ",
                                    nnz);

                            absolute_indices = indices_tensor.get_data<int64_t>();
                        }
                        return {
                            get_dense_tensor_as_constant(absolute_indices, values_tensor, shape)};
                    }
                    return {make_constant(node.get_attribute_value<Tensor>(attributes_names[0]))};
                }

            } // namespace set_13

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
