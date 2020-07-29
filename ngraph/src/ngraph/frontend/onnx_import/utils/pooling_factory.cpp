//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <iterator>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "utils/convpool.hpp"
#include "utils/pooling_factory.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace pooling
        {
            PoolingFactory::PoolingFactory(const Node& node)
                : m_onnx_node{node}
                , m_inputs{node.get_ng_inputs()}
                , m_strides{convpool::get_strides(node)}
                , m_dilations{convpool::get_dilations(node)}
                , m_auto_pad{convpool::get_auto_pad(node)}
            {
                const auto paddings = convpool::get_pads(node);
                const CoordinateDiff& padding_above{paddings.second};
                const CoordinateDiff& padding_below{paddings.first};
                m_padding_below = Shape{std::begin(padding_below), std::end(padding_below)};
                m_padding_above = Shape{std::begin(padding_above), std::end(padding_above)};
            }

            OutputVector PoolingFactory::make_avg_pool() const
            {
                const bool count_include_pad =
                    m_onnx_node.get_attribute_value<std::int64_t>("count_include_pad", 0);
                return {std::make_shared<default_opset::AvgPool>(m_inputs.at(0),
                                                                 m_strides,
                                                                 m_padding_below,
                                                                 m_padding_above,
                                                                 m_kernel_shape,
                                                                 !count_include_pad,
                                                                 op::RoundingType::FLOOR,
                                                                 m_auto_pad)};
            }

            OutputVector PoolingFactory::make_max_pool() const
            {
                return {std::make_shared<default_opset::MaxPool>(m_inputs.at(0),
                                                                 m_strides,
                                                                 m_padding_below,
                                                                 m_padding_above,
                                                                 m_kernel_shape,
                                                                 op::RoundingType::FLOOR,
                                                                 m_auto_pad)};
            }

            LocalPoolingFactory::LocalPoolingFactory(const Node& node)
                : PoolingFactory(node)
            {
                // Kernel shape is required
                m_kernel_shape = node.get_attribute_value<std::vector<std::size_t>>("kernel_shape");
            }

            GlobalPoolingFactory::GlobalPoolingFactory(const Node& node)
                : PoolingFactory(node)
            {
                const auto data_shape = node.get_ng_inputs().at(0).get_partial_shape();
                const auto data_rank = data_shape.rank();
                CHECK_VALID_NODE(
                    node, data_rank.is_static(), "Data rank must be static for global pooling ops");
                Shape kernel_shape;
                for (auto i = 2; i < data_rank.get_length(); ++i)
                {
                    CHECK_VALID_NODE(node,
                                     data_shape[i].is_static(),
                                     "All spatial dimensions must be known for global pooling ops");
                    kernel_shape.emplace_back(data_shape[i].get_length());
                }

                // Set shape to all but {N,C} axes.
                m_kernel_shape = kernel_shape;
            }
        } // namespace pooling
    }     // namespace onnx_import
} // namespace ngraph
