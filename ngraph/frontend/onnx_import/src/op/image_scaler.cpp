// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/image_scaler.hpp"
#include "default_opset.hpp"

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
            } // namespace set_1
        }     // namespace op
    }         // namespace onnx_import
} // namespace ngraph
