// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "op/cum_sum.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector cum_sum(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();
                    auto data = inputs.at(0);
                    bool exclusive = node.get_attribute_value<std::int64_t>("exclusive", 0);
                    bool reverse = node.get_attribute_value<std::int64_t>("reverse", 0);
                    Output<ngraph::Node> axis;

                    if (inputs.size() > 1)
                    {
                        axis = inputs.at(1); // optional input, 0-D tensor
                    }
                    else
                    {
                        axis =
                            default_opset::Constant::create(element::i64, Shape{}, {0}); // default
                    }
                    return OutputVector{
                        std::make_shared<default_opset::CumSum>(data, axis, exclusive, reverse)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
