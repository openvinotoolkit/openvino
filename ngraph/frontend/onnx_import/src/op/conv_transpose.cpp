// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/output_vector.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"
#include "op/conv_transpose.hpp"
#include "utils/convpool.hpp"

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
                    Output<ngraph::Node>
                        make_group_conv_backprop(const Output<ngraph::Node>& data,
                                                 const Output<ngraph::Node>& filters,
                                                 const Strides& strides,
                                                 const Strides& dilations,
                                                 const CoordinateDiff& pads_begin,
                                                 const CoordinateDiff& pads_end,
                                                 const ngraph::op::PadType& auto_pad_type,
                                                 const std::vector<std::int64_t>& output_shape,
                                                 const std::vector<std::int64_t>& output_padding)
                    {
                        if (output_shape.empty())
                        {
                            return std::make_shared<default_opset::GroupConvolutionBackpropData>(
                                data,
                                filters,
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad_type,
                                CoordinateDiff(std::begin(output_padding),
                                               std::end(output_padding)));
                        }
                        else
                        {
                            return std::make_shared<default_opset::GroupConvolutionBackpropData>(
                                data,
                                filters,
                                default_opset::Constant::create(
                                    element::i64, Shape{output_shape.size()}, output_shape),
                                strides,
                                dilations,
                                auto_pad_type,
                                CoordinateDiff(std::begin(output_padding),
                                               std::end(output_padding)));
                        }
                    }

                    Output<ngraph::Node>
                        make_conv_backprop(const Output<ngraph::Node>& data,
                                           const Output<ngraph::Node>& filters,
                                           const Strides& strides,
                                           const Strides& dilations,
                                           const CoordinateDiff& pads_begin,
                                           const CoordinateDiff& pads_end,
                                           const ngraph::op::PadType& auto_pad_type,
                                           const std::vector<std::int64_t>& output_shape,
                                           const std::vector<std::int64_t>& output_padding)
                    {
                        if (output_shape.empty())
                        {
                            return std::make_shared<default_opset::ConvolutionBackpropData>(
                                data,
                                filters,
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad_type,
                                CoordinateDiff(std::begin(output_padding),
                                               std::end(output_padding)));
                        }
                        else
                        {
                            return std::make_shared<default_opset::ConvolutionBackpropData>(
                                data,
                                filters,
                                default_opset::Constant::create(
                                    element::i64, Shape{output_shape.size()}, output_shape),
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad_type,
                                CoordinateDiff(std::begin(output_padding),
                                               std::end(output_padding)));
                        }
                    }

                    Output<ngraph::Node> get_prepared_bias(const Output<ngraph::Node>& bias,
                                                           const Output<ngraph::Node>& conv)
                    {
                        // Prepare bias shape [1, C, 1, 1]
                        const auto& conv_pshape = conv.get_partial_shape();
                        std::shared_ptr<ngraph::Node> bias_shape_node;

                        if (conv_pshape.rank().is_static() && conv_pshape[1].is_static())
                        {
                            Shape new_bias_shape(conv_pshape.rank().get_length(), 1);
                            new_bias_shape[1] = conv_pshape[1].get_length();

                            bias_shape_node = default_opset::Constant::create(
                                element::i64, Shape{new_bias_shape.size()}, new_bias_shape);
                        }
                        else
                        {
                            const auto conv_shape = std::make_shared<default_opset::ShapeOf>(conv);
                            const auto conv_rank =
                                std::make_shared<default_opset::ShapeOf>(conv_shape);

                            // Prepare new bias shape base: [1, 1, 1, 1, ... ]
                            const auto one_node =
                                default_opset::Constant::create(element::i64, Shape{1}, {1});
                            const auto two_node =
                                default_opset::Constant::create(element::i64, Shape{1}, {2});
                            const auto remaining_shape_length =
                                std::make_shared<default_opset::Subtract>(conv_rank, two_node);
                            const auto remaining_bias_shape_ones =
                                std::make_shared<default_opset::Broadcast>(one_node,
                                                                           remaining_shape_length);

                            const auto C_dim = std::make_shared<default_opset::StridedSlice>(
                                conv_shape,
                                one_node,                 // begin
                                two_node,                 // end
                                std::vector<int64_t>{0},  // begin mask
                                std::vector<int64_t>{0}); // end mask

                            // Construct new bias shape: [1, C, 1, 1, ... ]
                            bias_shape_node = std::make_shared<default_opset::Concat>(
                                OutputVector{one_node, C_dim, remaining_bias_shape_ones}, 0);
                        }

                        return std::make_shared<default_opset::Reshape>(
                                   bias, bias_shape_node, false)
                            ->add_provenance_group_members_above({bias});
                    }
                } // namespace

                OutputVector conv_transpose(const Node& node)
                {
                    const OutputVector& inputs = node.get_ng_inputs();

                    CHECK_VALID_NODE(node,
                                     inputs.size() == 2 || inputs.size() == 3,
                                     "Provided number of inputs is incorrect. The ConvTranspose "
                                     "operator expects 2 or 3 inputs.");

                    auto data = inputs[0];
                    auto filters = inputs[1];

                    const auto& data_pshape = data.get_partial_shape();
                    const auto& filters_pshape = filters.get_partial_shape();

                    std::size_t num_spatial_dims = 0;
                    Strides strides, dilations;
                    std::pair<CoordinateDiff, CoordinateDiff> paddings;
                    ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);

                    // Get attirbutes or infer them from input data rank it it's static.
                    if (data_pshape.rank().is_static())
                    {
                        num_spatial_dims = data_pshape.rank().get_length() - 2;
                    }
                    else if (filters_pshape.rank().is_static())
                    {
                        num_spatial_dims = filters_pshape.rank().get_length() - 2;
                    }
                    // Otherwise read "kernel_shape" attribute
                    else
                    {
                        CHECK_VALID_NODE(node,
                                         node.has_attribute("kernel_shape"),
                                         "\"kernel_shape\" attribute is required if data and "
                                         "filter inputs' ranks are dynamic.");
                        std::vector<std::size_t> kernel_shape =
                            node.get_attribute_value<std::vector<std::size_t>>("kernel_shape");

                        num_spatial_dims = kernel_shape.size();
                    }

                    strides = convpool::get_strides(node, num_spatial_dims);
                    dilations = convpool::get_dilations(node, num_spatial_dims);
                    paddings = convpool::get_pads(node, num_spatial_dims);
                    CoordinateDiff pads_begin = paddings.first;
                    CoordinateDiff pads_end = paddings.second;

                    std::vector<std::int64_t> output_shape{
                        node.get_attribute_value<std::vector<std::int64_t>>("output_shape", {})};

                    std::vector<std::int64_t> output_padding{
                        node.get_attribute_value<std::vector<std::int64_t>>(
                            "output_padding", std::vector<std::int64_t>(num_spatial_dims, 0))};

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    CHECK_VALID_NODE(
                        node, groups >= 0, "Incorrect value of 'group' attribute: ", groups);

                    Output<ngraph::Node> conv_node;

                    if (groups > 1)
                    {
                        filters = convpool::get_reshaped_filters(filters, groups);
                        conv_node = make_group_conv_backprop(data,
                                                             filters,
                                                             strides,
                                                             dilations,
                                                             pads_begin,
                                                             pads_end,
                                                             auto_pad_type,
                                                             output_shape,
                                                             output_padding);
                    }
                    else
                    {
                        conv_node = make_conv_backprop(data,
                                                       filters,
                                                       strides,
                                                       dilations,
                                                       pads_begin,
                                                       pads_end,
                                                       auto_pad_type,
                                                       output_shape,
                                                       output_padding);
                    }

                    // no bias param
                    if (inputs.size() < 3)
                    {
                        return {conv_node};
                    }
                    const auto reshaped_bias = get_prepared_bias(inputs[2], conv_node);

                    return {std::make_shared<default_opset::Add>(conv_node, reshaped_bias)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
