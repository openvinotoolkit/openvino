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

#include "ngraph/coordinate_diff.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/utils/convpool.hpp"
#include "onnx_import/utils/pooling_factory.hpp"

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
                , m_rounding_type{convpool::get_rounding_type(node)}
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
                                                                 m_rounding_type,
                                                                 m_auto_pad)};
            }

            OutputVector PoolingFactory::make_max_pool() const
            {
                return {std::make_shared<default_opset::MaxPool>(m_inputs.at(0),
                                                                 m_strides,
                                                                 m_padding_below,
                                                                 m_padding_above,
                                                                 m_kernel_shape,
                                                                 m_rounding_type,
                                                                 m_auto_pad)};
            }

            LocalPoolingFactory::LocalPoolingFactory(const Node& node)
                : PoolingFactory(node)
            {
                // Kernel shape is required
                m_kernel_shape = node.get_attribute_value<std::vector<std::size_t>>("kernel_shape");
            }
        } // namespace pooling
    }     // namespace onnx_import
} // namespace ngraph
