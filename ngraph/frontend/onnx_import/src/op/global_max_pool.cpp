// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "op/global_max_pool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector global_max_pool(const Node& node)
                {
                    // Generate axes for reduce operation which contain all spatial dims indexes.
                    // Examples:
                    // Input shape: [N, C, H, W]
                    // Input spatial dimensions are H and W
                    // Expected spatial dims indexes: [2, 3]
                    //
                    // Input shape: [N, C, H, W, D]
                    // Input spatial dimensions are H, W and D
                    // Expected spatial dims indexes: [2, 3, 4]
                    auto data = node.get_ng_inputs()[0];

                    const auto zero_node =
                        default_opset::Constant::create(element::i64, Shape{}, {0});
                    const auto one_node =
                        default_opset::Constant::create(element::i64, Shape{}, {1});
                    const auto two_node =
                        default_opset::Constant::create(element::i64, Shape{}, {2});

                    const auto data_shape = std::make_shared<default_opset::ShapeOf>(data);
                    const auto data_rank = std::make_shared<default_opset::ShapeOf>(data_shape);
                    const auto data_rank_as_scalar =
                        std::make_shared<default_opset::Squeeze>(data_rank);

                    const auto reduce_axes = std::make_shared<default_opset::Range>(
                        two_node, data_rank_as_scalar, one_node, element::i64);

                    return {std::make_shared<default_opset::ReduceMax>(data, reduce_axes, true)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
