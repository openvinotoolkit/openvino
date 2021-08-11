// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/conv_integer.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "utils/convpool.hpp"

using namespace ov::builder;

namespace ov
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector conv_integer(const Node& node)
                {
                    const OutputVector& inputs = node.get_ng_inputs();
                    auto num_inputs = inputs.size();
                    auto input = inputs.at(0);
                    auto filters = inputs.at(1);

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};
                    CHECK_VALID_NODE(
                        node,
                        groups == 1,
                        "Only value of 1 for 'group' supported for ConvInteger. Given: ",
                        groups);

                    auto window_movement_strides = convpool::get_strides(node);
                    auto window_dilation_strides = convpool::get_dilations(node);
                    auto paddings = convpool::get_pads(node);
                    ov::op::PadType auto_pad_type = convpool::get_auto_pad(node);
                    auto& padding_below = paddings.first;
                    auto& padding_above = paddings.second;
                    convpool::calculate_auto_pads(input.get_shape(),
                                                  filters.get_shape(),
                                                  window_movement_strides,
                                                  window_dilation_strides,
                                                  auto_pad_type,
                                                  padding_below,
                                                  padding_above);

                    const Strides default_data_dilation_strides(input.get_shape().size() - 2, 1);
                    auto scale_one = make_constant(ov::element::f32, Shape{}, 1);
                    auto input_zero_point = make_constant(input.get_element_type(), Shape{}, 0);
                    auto filters_zero_point = make_constant(filters.get_element_type(), Shape{}, 0);
                    auto output_zero_point = make_constant(ov::element::i32, Shape{}, 0);

                    if (num_inputs == 2)
                    {
                        return {std::make_shared<ov::opset0::QuantizedConvolution>(
                            input,
                            filters,
                            window_movement_strides,
                            window_dilation_strides,
                            padding_below,
                            padding_above,
                            default_data_dilation_strides,
                            scale_one,
                            input_zero_point,
                            scale_one,
                            filters_zero_point,
                            scale_one,
                            output_zero_point,
                            ov::element::i32,
                            ov::AxisSet{},
                            ov::AxisSet{},
                            ov::AxisSet{})};
                    }

                    input_zero_point = inputs.at(2);
                    if (num_inputs == 4)
                    {
                        filters_zero_point = inputs.at(3);
                    }

                    return {std::make_shared<ov::opset0::QuantizedConvolution>(
                        input,
                        filters,
                        window_movement_strides,
                        window_dilation_strides,
                        padding_below,
                        padding_above,
                        default_data_dilation_strides,
                        scale_one,
                        input_zero_point,
                        scale_one,
                        filters_zero_point,
                        scale_one,
                        output_zero_point,
                        ov::element::i32,
                        ov::AxisSet{},
                        ov::AxisSet{},
                        ov::AxisSet{})};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ov
