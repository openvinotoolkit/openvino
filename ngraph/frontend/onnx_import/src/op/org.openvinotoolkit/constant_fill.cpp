// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector constant_fill(const Node& node)
                {
                    const auto target_shape = node.get_ng_inputs().at(0);
                    const auto fill_value = node.get_attribute_value<float>("fill_value", 0.f);
                    const auto input_as_shape =
                        node.get_attribute_value<float>("input_as_shape", 1.f);
                    CHECK_VALID_NODE(node,
                                     input_as_shape == 1,
                                     "Only input_as_shape=1 is supported by ConstantFill");

                    const auto const_val_to_fill =
                        default_opset::Constant::create(element::f32, {}, {fill_value});
                    return {std::make_shared<default_opset::Broadcast>(const_val_to_fill,
                                                                       target_shape)};
                }

            } // namespace set_1
        }     // namespace op
    }         // namespace onnx_import

} // namespace ngraph
