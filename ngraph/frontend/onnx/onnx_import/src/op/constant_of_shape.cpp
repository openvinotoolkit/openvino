// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/tensor.hpp"
#include "default_opset.hpp"
#include "ngraph/op/constant.hpp"
#include "op/constant.hpp"
#include "utils/reshape.hpp"

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
