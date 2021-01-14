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

#include "ngraph/op/constant.hpp"
#include "onnx_import/core/tensor.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/constant.hpp"
#include "onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector constant_of_shape(const onnx_import::Node& node)
                {
                    Output<ngraph::Node> constant_value;
                    if (node.has_attribute("value"))
                    {
                        auto value_tensor = node.get_attribute_value<Tensor>("value");
                        constant_value = value_tensor.get_ng_constant();
                        constant_value = reshape::interpret_as_scalar(constant_value);
                    }
                    else
                    {
                        constant_value = default_opset::Constant::create(element::f32, {}, {0});
                    }
                    return {std::make_shared<default_opset::Broadcast>(constant_value,
                                                                       node.get_ng_inputs().at(0))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
