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

#include "onnx_import/op/image_scaler.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector image_scaler(const Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    NGRAPH_CHECK(
                        inputs.size() == 1, "ImageScaler 1 input tensor. Got: ", inputs.size());

                    const auto data = inputs[0];
                    const auto& data_shape = data.get_partial_shape();
                    NGRAPH_CHECK(data_shape.rank().same_scheme({4}),
                                 "ImageScaler expects a 4D tensor with NCHW format. Got: ",
                                 data_shape);

                    const auto scale = node.get_attribute_value<float>("scale", 1.0);
                    const auto bias = node.get_attribute_value<std::vector<float>>("bias");

                    NGRAPH_CHECK(data_shape[1].same_scheme(bias.size()),
                                 "Number of bias attribute elements: ",
                                 bias.size(),
                                 " does not match the channel dimension: ",
                                 data_shape[1].get_length());

                    const auto scale_const =
                        default_opset::Constant::create(data.get_element_type(), Shape{}, {scale});

                    const auto bias_const = default_opset::Constant::create(
                        data.get_element_type(), {1, bias.size(), 1, 1}, bias);

                    const auto scaler = std::make_shared<default_opset::Add>(
                        std::make_shared<default_opset::Multiply>(data, scale_const), bias_const);

                    return {scaler};
                }
            }
        }
    }
}
