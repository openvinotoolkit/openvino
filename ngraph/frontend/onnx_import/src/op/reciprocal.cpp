// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/op/constant.hpp"
#include "op/reciprocal.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector reciprocal(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);

                    auto one_node =
                        default_opset::Constant::create(data.get_element_type(), Shape{}, {1});
                    return {std::make_shared<default_opset::Divide>(one_node, data)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
