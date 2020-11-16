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

#include <cstddef> // std::size_t
#include <vector>

#include "ngraph/op/constant.hpp"
#include "onnx_import/default_opset.hpp"
#include "reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            namespace
            {
                std::shared_ptr<default_opset::Constant> get_reduction_axes(const Node& node)
                {
                    auto reduction_axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

                    if (reduction_axes.empty())
                    {
                        const auto input_rank =
                            node.get_ng_inputs().at(0).get_partial_shape().rank();

                        NGRAPH_CHECK(input_rank.is_static(),
                                     "The input tensor's rank needs to be known(static) when the "
                                     "'axes' attribute is not specified. Node: ",
                                     node.get_description());

                        reduction_axes = onnx_import::common::get_monotonic_range<int64_t>(
                            input_rank.get_length());
                    }

                    return default_opset::Constant::create(
                        element::i64, Shape{reduction_axes.size()}, reduction_axes);
                }
            } // namespace

            std::shared_ptr<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const Output<ngraph::Node>& ng_input,
                                     ReductionOpProvider reduction_function)
            {
                const auto reduction_axes = get_reduction_axes(node);
                const std::int64_t keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);

                const auto op_node =
                    reduction_function(ng_input, reduction_axes, static_cast<bool>(keepdims));

                return op_node;
            }

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph
