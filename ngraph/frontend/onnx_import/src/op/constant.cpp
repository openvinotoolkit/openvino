//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "onnx_import/op/constant.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "onnx_import/core/tensor.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
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
                            NGRAPH_WARN << "Could not create an nGraph Constant for an ONNX "
                                           "Constant node. Detailed error:\n"
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

                    inline std::shared_ptr<default_opset::Constant>
                        make_constant(const Tensor& tensor)
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
                        default: throw error::tensor::invalid_data_type{tensor};
                        }
                    }
                }

                OutputVector constant(const onnx_import::Node& node)
                {
                    return {make_constant(node.get_attribute_value<Tensor>("value"))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
