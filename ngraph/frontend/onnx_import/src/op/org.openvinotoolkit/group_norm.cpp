//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "onnx_import/op/org.openvinotoolkit/group_norm.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/utils/common.hpp"
#include "onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace detail
            {
                namespace
                {
                    // This function creates a shape to which we need to reshape the input
                    // before normalization.
                    // If data shape is [N,C,H,W], the function returns
                    // [N, num_groups, C // num_groups, H, W]
                    std::shared_ptr<ngraph::Node>
                        create_group_norm_shape(const Output<ngraph::Node>& data, size_t num_groups)
                    {
                        const auto& pshape = data.get_partial_shape();
                        NGRAPH_CHECK(pshape.rank().is_static());
                        size_t rank_size = pshape.rank().get_length();
                        NGRAPH_CHECK(rank_size >= 3, "3-D and above tensors supported only");

                        if (pshape.is_static())
                        {
                            const auto& shape = pshape.to_shape();
                            std::vector<size_t> new_shape{
                                shape[0], num_groups, shape[1] / num_groups};
                            for (size_t i = 2; i < rank_size; i++)
                            {
                                new_shape.push_back(shape[i]);
                            }
                            return default_opset::Constant::create(
                                element::i64, Shape{new_shape.size()}, new_shape);
                        }

                        auto shape = std::make_shared<default_opset::ShapeOf>(data);
                        auto splits = builder::opset1::split(shape, rank_size);
                        auto num_groups_const =
                            default_opset::Constant::create(element::i64, Shape{1}, {num_groups});
                        NodeVector new_shape{
                            splits[0].get_node_shared_ptr(),
                            num_groups_const,
                            std::make_shared<default_opset::Divide>(splits[1], num_groups_const)};
                        for (size_t i = 2; i < rank_size; i++)
                        {
                            new_shape.push_back(splits[i].get_node_shared_ptr());
                        }
                        return std::make_shared<default_opset::Concat>(new_shape, 0);
                    }
                }
            } // detail

            namespace set_1
            {
                OutputVector group_norm(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();
                    NGRAPH_CHECK(inputs.size() == 3,
                                 "Invalid number of inputs. Expected 3, actual " +
                                     std::to_string(inputs.size()));

                    auto data = inputs[0];
                    auto scale = inputs[1];
                    auto bias = inputs[2];

                    size_t num_groups =
                        static_cast<size_t>(node.get_attribute_value<int64_t>("num_groups"));
                    float eps = node.get_attribute_value<float>("eps", 1e-5);

                    auto data_pshape = data.get_partial_shape();
                    std::shared_ptr<ngraph::Node> data_shape_node;
                    if (data_pshape.is_static())
                    {
                        auto shape = data_pshape.to_shape();
                        data_shape_node = default_opset::Constant::create(
                            element::u64, Shape{shape.size()}, shape);
                    }
                    else
                    {
                        data_shape_node = std::make_shared<default_opset::ShapeOf>(data);
                    }
                    auto data_reshaped = std::make_shared<default_opset::Reshape>(
                        data, detail::create_group_norm_shape(data, num_groups), true);
                    const auto reduction_axes =
                        common::get_monotonic_range_along_node_rank(data_reshaped, 2);
                    auto mean = std::make_shared<default_opset::ReduceMean>(
                        data_reshaped, reduction_axes, true);
                    auto diff = std::make_shared<default_opset::Subtract>(data_reshaped, mean);
                    auto variance = std::make_shared<default_opset::ReduceMean>(
                        std::make_shared<default_opset::Power>(
                            diff, default_opset::Constant::create(element::f32, Shape{}, {2})),
                        reduction_axes,
                        true);

                    const std::shared_ptr<ngraph::Node> eps_node =
                        std::make_shared<default_opset::Constant>(element::f32, Shape{}, eps);
                    const auto sqrt = std::make_shared<default_opset::Sqrt>(
                        std::make_shared<default_opset::Add>(variance, eps_node));

                    const auto& rank = data.get_partial_shape().rank();
                    NGRAPH_CHECK(rank.is_static());
                    auto data_rank_size = rank.get_length();

                    std::shared_ptr<ngraph::Node> result =
                        std::make_shared<default_opset::Divide>(diff, sqrt);
                    result =
                        std::make_shared<default_opset::Reshape>(result, data_shape_node, true);
                    result = std::make_shared<default_opset::Multiply>(
                        reshape::reshape_channel_shaped_node_to_nchw(scale, data_rank_size),
                        result);
                    result = std::make_shared<default_opset::Add>(
                        result, reshape::reshape_channel_shaped_node_to_nchw(bias, data_rank_size));

                    return {result};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
