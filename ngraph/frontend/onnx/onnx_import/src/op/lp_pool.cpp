// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/util.hpp"
#include "op/lp_pool.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector global_lp_pool(const Node& node)
                {
                    const Output<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const std::size_t channel_axis{1};

                    const auto data_shape = data.get_partial_shape();
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

                    CHECK_VALID_NODE(
                        node,
                        p_norm >= 0,
                        "Only positive (including zero) values are supported for 'p' attribute.");

                    OutputVector slices =
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
