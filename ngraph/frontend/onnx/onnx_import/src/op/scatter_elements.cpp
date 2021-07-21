// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "op/scatter_elements.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector scatter_elements(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto indices = node.get_ng_inputs().at(1);
                    const auto updates = node.get_ng_inputs().at(2);

                    const auto axis = node.get_attribute_value<std::int64_t>("axis", 0);
                    const auto axis_node =
                        default_opset::Constant::create(element::i64, Shape{}, {axis});

                    return {std::make_shared<ngraph::opset3::ScatterElementsUpdate>(
                        data, indices, updates, axis_node)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
