// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/squeeze.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"
#include "op/squeeze.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector squeeze(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    std::vector<std::int64_t> axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});
                    const auto data_rank = data.get_partial_shape().rank();

                    std::vector<std::size_t> normalized_axes =
                        ngraph::normalize_axes(node.get_description(), axes, data_rank);
                    auto axes_node = std::make_shared<default_opset::Constant>(
                        element::u64, Shape{normalized_axes.size()}, normalized_axes);

                    return {std::make_shared<default_opset::Squeeze>(data, axes_node)};
                }

            } // namespace set_1

            namespace set_13
            {
                OutputVector squeeze(const Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    if (inputs.size() < 2)
                    {
                        return {std::make_shared<default_opset::Squeeze>(inputs.at(0))};
                    }
                    else
                    {
                        return {
                            std::make_shared<default_opset::Squeeze>(inputs.at(0), inputs.at(1))};
                    }
                }

            } // namespace set_13
        }     // namespace op
    }         // namespace onnx_import
} // namespace ngraph
