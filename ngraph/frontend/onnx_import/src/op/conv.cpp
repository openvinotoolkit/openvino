// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "op/conv.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                namespace
                {
                    std::shared_ptr<ngraph::op::Op>
                        make_ng_convolution(const Output<ngraph::Node>& data,
                                            const Output<ngraph::Node>& filters,
                                            const ngraph::Strides& strides,
                                            const ngraph::Strides& dilations,
                                            const ngraph::CoordinateDiff& padding_below,
                                            const ngraph::CoordinateDiff& padding_above,
                                            int64_t groups,
                                            const ngraph::op::PadType& auto_pad)
                    {
                        if (groups > 1)
                        {
                            const auto reshaped_filters =
                                convpool::get_reshaped_filters(filters, groups);

                            return std::make_shared<default_opset::GroupConvolution>(
                                data,
                                reshaped_filters,
                                strides,
                                padding_below,
                                padding_above,
                                dilations,
                                auto_pad);
                        }
                        else
                        {
                            return std::make_shared<default_opset::Convolution>(data,
                                                                                filters,
                                                                                strides,
                                                                                padding_below,
                                                                                padding_above,
                                                                                dilations,
                                                                                auto_pad);
                        }
                    }

                    std::shared_ptr<ngraph::Node> add_bias(const Output<ngraph::Node>& ng_conv,
                                                           const Output<ngraph::Node>& bias)
                    {
                        const auto conv_shape = std::make_shared<default_opset::ShapeOf>(ng_conv);
                        const auto conv_rank = std::make_shared<default_opset::ShapeOf>(conv_shape);

                        return {std::make_shared<default_opset::Add>(
                            ng_conv,
                            reshape::reshape_channel_shaped_node_to_nchw(bias, conv_rank))};
                    }
                } // namespace

                OutputVector conv(const Node& node)
                {
                    // in the current implementation we assume that the data input rank is static
                    // and only the 'batch' dimension can be dynamic
                    const OutputVector& inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);
                    const auto filters = inputs.at(1);
                    const auto groups = node.get_attribute_value<int64_t>("group", 1);

                    NGRAPH_CHECK(data.get_partial_shape().rank().is_static(),
                                 "The input data tensor's rank has to be known (static)");

                    const auto strides = convpool::get_strides(node);
                    const auto dilations = convpool::get_dilations(node);
                    const auto paddings = convpool::get_pads(node);
                    const ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);
                    const auto& padding_below = paddings.first;
                    const auto& padding_above = paddings.second;

                    const auto conv_node = make_ng_convolution(data,
                                                               filters,
                                                               strides,
                                                               dilations,
                                                               padding_below,
                                                               padding_above,
                                                               groups,
                                                               auto_pad_type);

                    // no bias param
                    if (inputs.size() < 3)
                    {
                        return {conv_node};
                    }
                    else
                    {
                        const auto& bias = inputs.at(2);
                        const auto& bias_ps = bias.get_partial_shape();

                        NGRAPH_CHECK(bias_ps.rank().is_static() && bias_ps.rank().get_length() == 1,
                                     "The bias input needs to be 1D vector");

                        return {add_bias(conv_node, bias)};
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
