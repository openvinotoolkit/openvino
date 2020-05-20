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

#include <cstddef>
#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "lp_pool.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/util.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector global_lp_pool(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const std::size_t channel_axis{1};

                    const auto data_shape = data->get_output_partial_shape(0);
                    NGRAPH_CHECK(data_shape.rank().is_static(),
                                 "Rank of input data must be static");
                    NGRAPH_CHECK(data_shape.rank().get_length() >= 2,
                                 "Rank of input data must be greater or equal to 2");
                    NGRAPH_CHECK(data_shape[0].is_static(),
                                 "First dimension of input data must be static");
                    NGRAPH_CHECK(data_shape[channel_axis].is_static(),
                                 "Channel dimension of intput data must be static");

                    const std::size_t channels_count = data_shape[channel_axis].get_length();
                    const std::int64_t p_norm{node.get_attribute_value<std::int64_t>("p", 2)};

                    ASSERT_VALID_ARGUMENT(node, p_norm >= 0)
                        << "Only positive (including zero) values are supported for 'p' attribute.";

                    NodeVector slices =
                        ngraph::builder::opset1::split(data, channels_count, channel_axis);

                    for (auto& slice : slices)
                    {
                        // all dimensions except spatial/feature
                        const auto reduction_axes =
                            common::get_monotonic_range_along_node_rank(data, 2);

                        slice = ngraph::builder::opset1::lp_norm(
                            slice, reduction_axes, static_cast<std::size_t>(p_norm));

                        // output shape is all ones except N channel
                        Shape output_shape(data_shape.rank().get_length(), 1);
                        output_shape.at(0) = data_shape[0].get_length();

                        const auto reshape_pattern = default_opset::Constant::create(
                            element::i64, Shape{output_shape.size()}, output_shape);

                        slice =
                            std::make_shared<default_opset::Reshape>(slice, reshape_pattern, false);
                    }

                    return {std::make_shared<default_opset::Concat>(slices, channel_axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
