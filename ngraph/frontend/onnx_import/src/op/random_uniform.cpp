// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/random_uniform.hpp"
#include <onnx/onnx_pb.h>
#include "default_opset.hpp"

#include <random>

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace detail
            {
                namespace
                {
                    template <typename T,
                              typename std::enable_if<std::is_floating_point<T>::value,
                                                      bool>::type = true>
                    OutputVector
                        gen_random_constant(const Shape& shape, float low, float high, float seed)
                    {
                        std::mt19937 gen(seed);
                        std::uniform_real_distribution<T> dis(low, high);
                        size_t n = shape_size(shape);
                        std::vector<T> values;
                        values.reserve(n);
                        for (size_t i = 0; i < n; i++)
                            values.push_back(dis(gen));
                        return {default_opset::Constant::create(element::from<T>(), shape, values)};
                    }

                    template <
                        typename T,
                        typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
                    OutputVector
                        gen_random_constant(const Shape& shape, float low, float high, float seed)
                    {
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<T> dis(low, high);
                        size_t n = shape_size(shape);
                        std::vector<T> values;
                        values.reserve(n);
                        for (size_t i = 0; i < n; i++)
                            values.push_back(dis(gen));
                        return {default_opset::Constant::create(element::from<T>(), shape, values)};
                    }
                } // namespace
            }     // namespace detail
            namespace set_1
            {
                OutputVector random_uniform(const Node& node)
                {
                    float low = node.get_attribute_value<float>("low", 0.0f);
                    float high = node.get_attribute_value<float>("high", 1.0f);
                    float seed = node.get_attribute_value<float>("seed", std::random_device()());

                    auto shape_attr = node.get_attribute_value<std::vector<int64_t>>("shape");
                    Shape shape;
                    std::transform(shape_attr.begin(),
                                   shape_attr.end(),
                                   std::back_inserter(shape),
                                   [](int64_t dim) -> size_t { return static_cast<size_t>(dim); });

                    using ONNX_NAMESPACE::TensorProto;
                    TensorProto::DataType dtype =
                        static_cast<TensorProto::DataType>(node.get_attribute_value<int64_t>(
                            "dtype", static_cast<int64_t>(TensorProto::FLOAT)));
                    switch (dtype)
                    {
                    case TensorProto::FLOAT:
                        return detail::gen_random_constant<float>(shape, low, high, seed);
                    case TensorProto::INT32:
                        return detail::gen_random_constant<int32_t>(shape, low, high, seed);
                    case TensorProto::UINT32:
                        return detail::gen_random_constant<uint32_t>(shape, low, high, seed);
                    case TensorProto::INT16:
                        return detail::gen_random_constant<int16_t>(shape, low, high, seed);
                    case TensorProto::UINT16:
                        return detail::gen_random_constant<uint16_t>(shape, low, high, seed);
                    case TensorProto::INT64:
                        return detail::gen_random_constant<int64_t>(shape, low, high, seed);
                    case TensorProto::UINT64:
                        return detail::gen_random_constant<uint64_t>(shape, low, high, seed);
                    case TensorProto::DOUBLE:
                        return detail::gen_random_constant<double>(shape, low, high, seed);
                    default: NGRAPH_CHECK(false, "Type not supported");
                    }

                    return {};
                }

            } // namespace set_1
        }     // namespace op

    } // namespace onnx_import

} // namespace ngraph
