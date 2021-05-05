// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include <cstddef>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/quantization/quantized_linear_convolution.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/strides.hpp"
#include "op/quant_conv.hpp"

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
                    struct OpScale
                    {
                        Output<ngraph::Node> data_scale;
                        Output<ngraph::Node> filter_scale;
                        Output<ngraph::Node> output_scale;
                    };

                    struct OpZeroPoint
                    {
                        Output<ngraph::Node> data_zero_point;
                        Output<ngraph::Node> filter_zero_point;
                        Output<ngraph::Node> output_zero_point;
                    };

                    std::shared_ptr<ngraph::Node>
                        make_ng_quant_conv(const Output<ngraph::Node>& data,
                                           const Output<ngraph::Node>& filters,
                                           const Strides& strides,
                                           const Strides& filter_dilations,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilations,
                                           int groups,
                                           const OpScale& op_scale,
                                           const OpZeroPoint& op_zero_point,
                                           const Output<ngraph::Node>& bias = nullptr)
                    {
                        ngraph::element::Type output_type;
                        if (data.get_element_type() == ngraph::element::u8 &&
                            filters.get_element_type() == ngraph::element::i8)
                        {
                            output_type = ngraph::element::i8;
                        }
                        else if (data.get_element_type() == ngraph::element::u8 &&
                                 filters.get_element_type() == ngraph::element::u8)
                        {
                            output_type = ngraph::element::u8;
                        }
                        if (groups > 1)
                        {
                            // Split one convolution op to N ops where N is the number of groups
                            // and concat results after computation.
                            std::size_t n_data_channels{data.get_shape().at(1)};
                            std::size_t n_filters_channels{filters.get_shape().at(0)};

                            std::size_t data_group_size{n_data_channels / groups};
                            std::size_t filters_group_size{n_filters_channels / groups};
                            OutputVector convolution_nodes;

                            // initial bounds for splice
                            std::vector<std::size_t> data_lower_bounds(data.get_shape().size());
                            std::vector<std::size_t> data_upper_bounds{data.get_shape()};
                            std::vector<std::size_t> filters_lower_bounds(
                                filters->get_shape().size());
                            std::vector<std::size_t> filters_upper_bounds{filters.get_shape()};

                            for (int64_t group{0}; group < groups; ++group)
                            {
                                // slice data
                                data_lower_bounds[1] = group * data_group_size;
                                data_upper_bounds[1] = (group + 1) * data_group_size;
                                auto sliced_data = std::make_shared<ngraph::opset0::Slice>(
                                    data, data_lower_bounds, data_upper_bounds);
                                // slice filters
                                filters_lower_bounds[0] = group * filters_group_size;
                                filters_upper_bounds[0] = (group + 1) * filters_group_size;
                                auto sliced_filters = std::make_shared<ngraph::opset0::Slice>(
                                    filters, filters_lower_bounds, filters_upper_bounds);

                                if (bias.get_node())
                                {
                                    throw ngraph_error(
                                        "Groups != 1 not supported for Quantized Convolution with "
                                        "bias.");
                                }
                                else
                                {
                                    convolution_nodes.push_back(
                                        std::make_shared<ngraph::opset0::QuantizedConvolution>(
                                            sliced_data,
                                            sliced_filters,
                                            strides,
                                            filter_dilations,
                                            padding_below,
                                            padding_above,
                                            data_dilations,
                                            op_scale.data_scale,
                                            op_zero_point.data_zero_point,
                                            op_scale.filter_scale,
                                            op_zero_point.filter_zero_point,
                                            op_scale.output_scale,
                                            op_zero_point.output_zero_point,
                                            output_type,
                                            ngraph::AxisSet{},
                                            ngraph::AxisSet{},
                                            ngraph::AxisSet{}));
                                }
                            }
                            std::size_t concatenation_axis = 1;
                            return std::make_shared<default_opset::Concat>(convolution_nodes,
                                                                           concatenation_axis);
                        }
                        else
                        {
                            if (bias.get_node())
                            {
                                return ngraph::builder::quantization::
                                    QuantizedLinearConvolutionBias(data,
                                                                   filters,
                                                                   bias,
                                                                   strides,
                                                                   filter_dilations,
                                                                   padding_below,
                                                                   padding_above,
                                                                   data_dilations,
                                                                   op_scale.data_scale,
                                                                   op_scale.filter_scale,
                                                                   op_scale.output_scale);
                            }
                            else
                            {
                                return std::make_shared<ngraph::opset0::QuantizedConvolution>(
                                    data,
                                    filters,
                                    strides,
                                    filter_dilations,
                                    padding_below,
                                    padding_above,
                                    data_dilations,
                                    op_scale.data_scale,
                                    op_zero_point.data_zero_point,
                                    op_scale.filter_scale,
                                    op_zero_point.filter_zero_point,
                                    op_scale.output_scale,
                                    op_zero_point.output_zero_point,
                                    output_type,
                                    ngraph::AxisSet{},
                                    ngraph::AxisSet{},
                                    ngraph::AxisSet{});
                            }
                        }
                    }

                } // namespace

                OutputVector quant_conv(const Node& node)
                {
                    const OutputVector& inputs = node.get_ng_inputs();
                    auto data = inputs.at(0);
                    auto filters = inputs.at(3);

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

                    auto data_scale = inputs.at(1);
                    auto data_zero_point = inputs.at(2);
                    auto filters_scale = inputs.at(4);
                    auto filters_zero_point = inputs.at(5);
                    auto output_scale = inputs.at(6);
                    auto output_zero_point = inputs.at(7);

                    CHECK_VALID_NODE(node,
                                     ((groups >= 0) &&
                                      (groups <= static_cast<int64_t>(data.get_shape().at(1))) &&
                                      (groups <= static_cast<int64_t>(filters.get_shape().at(0)))),
                                     "incorrect value of 'group' attribute: ",
                                     groups);

                    std::size_t n_data_channels{data.get_shape().at(1)};
                    std::size_t n_filters_channels{filters.get_shape().at(0)};

                    CHECK_VALID_NODE(
                        node,
                        n_data_channels % groups == 0,
                        "provided group attribute value must be a multiple of data channels "
                        "count.");
                    CHECK_VALID_NODE(
                        node,
                        n_filters_channels % groups == 0,
                        "provided group attribute value must be a multiple of filter channels "
                        "count.");

                    Strides strides = convpool::get_strides(node);
                    Strides filter_dilations = convpool::get_dilations(node);
                    Strides data_dilations = Strides(convpool::get_kernel_shape(node).size(), 1UL);
                    auto paddings = convpool::get_pads(node);
                    ngraph::op::PadType auto_pad_type = convpool::get_auto_pad(node);
                    CoordinateDiff& padding_below = paddings.first;
                    CoordinateDiff& padding_above = paddings.second;
                    convpool::calculate_auto_pads(data.get_shape(),
                                                  filters.get_shape(),
                                                  strides,
                                                  filter_dilations,
                                                  auto_pad_type,
                                                  padding_below,
                                                  padding_above);

                    std::shared_ptr<ngraph::Node> conv_node = nullptr;

                    // no bias param
                    if (inputs.size() == 9 && !ngraph::op::is_null(inputs.at(8)))
                    {
                        auto bias = inputs.at(8);
                        conv_node = make_ng_quant_conv(
                            data,
                            filters,
                            strides,
                            filter_dilations,
                            padding_below,
                            padding_above,
                            data_dilations,
                            groups,
                            OpScale{data_scale, filters_scale, output_scale},
                            OpZeroPoint{data_zero_point, filters_zero_point, output_zero_point},
                            bias);
                    }
                    else
                    {
                        conv_node = make_ng_quant_conv(
                            data,
                            filters,
                            strides,
                            filter_dilations,
                            padding_below,
                            padding_above,
                            data_dilations,
                            groups,
                            OpScale{data_scale, filters_scale, output_scale},
                            OpZeroPoint{data_zero_point, filters_zero_point, output_zero_point});
                    }

                    return {conv_node};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
