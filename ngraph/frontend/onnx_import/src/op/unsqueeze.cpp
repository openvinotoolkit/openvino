// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"
#include "op/unsqueeze.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector unsqueeze(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {});
                    auto axes_node = std::make_shared<default_opset::Constant>(
                        element::i64, Shape{axes.size()}, axes);
                    return {std::make_shared<default_opset::Unsqueeze>(data, axes_node)};
                }

            } // namespace set_1

            namespace set_13
            {
                OutputVector unsqueeze(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();
                    return {std::make_shared<default_opset::Unsqueeze>(inputs.at(0), inputs.at(1))};
                }

            } // namespace set_13
        }     // namespace op

    } // namespace onnx_import

} // namespace ngraph
