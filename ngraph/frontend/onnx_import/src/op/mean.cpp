// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/mean.hpp"
#include "default_opset.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector mean(const Node& node)
                {
                    auto sum = variadic::make_ng_variadic_op<default_opset::Add>(node).front();
                    auto count = default_opset::Constant::create(
                        sum.get_element_type(), Shape{}, {node.get_ng_inputs().size()});

                    return {std::make_shared<default_opset::Divide>(sum, count)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
