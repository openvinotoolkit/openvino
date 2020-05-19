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

#include <memory>

#include "ngraph/builder/quantized_conv_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        shared_ptr<Node> QuantizedConvolutionBuilder(const Output<Node>& input,
                                                     const Output<Node>& filters,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const CoordinateDiff& padding_below,
                                                     const CoordinateDiff& padding_above,
                                                     const Strides& data_dilation_strides,
                                                     const Output<Node>& min_input,
                                                     const Output<Node>& max_input,
                                                     const Output<Node>& min_filter,
                                                     const Output<Node>& max_filter,
                                                     const Output<Node>& min_output,
                                                     const Output<Node>& max_output,
                                                     const ngraph::element::Type& output_type,
                                                     const ngraph::AxisSet& input_axes,
                                                     const ngraph::AxisSet& filter_axes,
                                                     const ngraph::AxisSet& output_axes)
        {
            auto input_scale =
                quantization_utils::get_scale(min_input, max_input, input.get_element_type());
            auto filter_scale =
                quantization_utils::get_scale(min_filter, max_filter, filters.get_element_type());
            auto output_scale = quantization_utils::get_scale(min_output, max_output, output_type);

            // TODO: Check for this later
            // For Builders the zero point is assumed to be zero (for now)
            auto input_zero_point = op::Constant::create(input.get_element_type(), Shape{}, {0});
            auto filter_zero_point = op::Constant::create(filters.get_element_type(), Shape{}, {0});

            return make_shared<op::QuantizedConvolution>(
                       input,
                       filters,
                       window_movement_strides,
                       window_dilation_strides,
                       padding_below,
                       padding_above,
                       data_dilation_strides,
                       input_scale,
                       input_zero_point,
                       filter_scale,
                       filter_zero_point,
                       output_scale,
                       filter_zero_point, // output type will be same as filter
                       output_type,
                       input_axes,
                       filter_axes,
                       output_axes)
                ->add_provenance_group_members_above({input,
                                                      filters,
                                                      min_input,
                                                      max_input,
                                                      min_filter,
                                                      max_filter,
                                                      min_output,
                                                      max_output});
        }

        shared_ptr<Node> QuantizedConvolutionBiasBuilder(const Output<Node>& input,
                                                         const Output<Node>& filters,
                                                         const Output<Node>& bias,
                                                         const Strides& window_movement_strides,
                                                         const Strides& window_dilation_strides,
                                                         const CoordinateDiff& padding_below,
                                                         const CoordinateDiff& padding_above,
                                                         const Strides& data_dilation_strides,
                                                         const Output<Node>& min_input,
                                                         const Output<Node>& max_input,
                                                         const Output<Node>& min_filter,
                                                         const Output<Node>& max_filter,
                                                         const Output<Node>& min_output,
                                                         const Output<Node>& max_output,
                                                         const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto input_scale =
                quantization_utils::get_scale(min_input, max_input, input.get_element_type());
            auto filter_scale =
                quantization_utils::get_scale(min_filter, max_filter, filters.get_element_type());
            auto output_scale = quantization_utils::get_scale(min_output, max_output, output_et);
            auto requantization_scale = input_scale * filter_scale / output_scale;

            auto mybias = bias;
            if (bias.get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input.get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale = quantization_utils::get_bias_scale(
                    min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }

            return make_shared<op::QuantizedConvolutionBias>(input,
                                                             filters,
                                                             mybias,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale,
                                                             with_relu)
                ->add_provenance_group_members_above({input,
                                                      filters,
                                                      bias,
                                                      min_input,
                                                      max_input,
                                                      min_filter,
                                                      max_filter,
                                                      min_output,
                                                      max_output});
        }

        shared_ptr<Node> QuantizedConvolutionReluBuilder(const Output<Node>& input,
                                                         const Output<Node>& filters,
                                                         const Strides& window_movement_strides,
                                                         const Strides& window_dilation_strides,
                                                         const CoordinateDiff& padding_below,
                                                         const CoordinateDiff& padding_above,
                                                         const Strides& data_dilation_strides,
                                                         const Output<Node>& min_input,
                                                         const Output<Node>& max_input,
                                                         const Output<Node>& min_filter,
                                                         const Output<Node>& max_filter,
                                                         const Output<Node>& min_output,
                                                         const Output<Node>& max_output)
        {
            auto input_scale =
                quantization_utils::get_scale(min_input, max_input, input.get_element_type());
            auto filter_scale =
                quantization_utils::get_scale(min_filter, max_filter, filters.get_element_type());
            auto output_scale = quantization_utils::get_scale(min_output, max_output, element::u8);
            auto requantization_scale = input_scale * filter_scale / output_scale;

            return make_shared<op::QuantizedConvolutionRelu>(input,
                                                             filters,
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             requantization_scale)
                ->add_provenance_group_members_above({input,
                                                      filters,
                                                      min_input,
                                                      max_input,
                                                      min_filter,
                                                      max_filter,
                                                      min_output,
                                                      max_output});
        }

        shared_ptr<Node> QuantizedConvolutionBiasAddBuilder(const Output<Node>& input,
                                                            const Output<Node>& filters,
                                                            const Output<Node>& bias,
                                                            const Output<Node>& sum_input,
                                                            const Strides& window_movement_strides,
                                                            const Strides& window_dilation_strides,
                                                            const CoordinateDiff& padding_below,
                                                            const CoordinateDiff& padding_above,
                                                            const Strides& data_dilation_strides,
                                                            const Output<Node>& min_input,
                                                            const Output<Node>& max_input,
                                                            const Output<Node>& min_filter,
                                                            const Output<Node>& max_filter,
                                                            const Output<Node>& min_output,
                                                            const Output<Node>& max_output,
                                                            const Output<Node>& min_sum_input,
                                                            const Output<Node>& max_sum_input,
                                                            const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto input_scale =
                quantization_utils::get_scale(min_input, max_input, input.get_element_type());
            auto filter_scale =
                quantization_utils::get_scale(min_filter, max_filter, filters.get_element_type());
            auto output_scale = quantization_utils::get_scale(min_output, max_output, output_et);
            auto requantization_scale = input_scale * filter_scale / output_scale;

            auto sum_scale = builder::quantization_utils::get_sum_scale(
                min_output, max_output, min_sum_input, max_sum_input);

            auto mybias = bias;
            if (bias.get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input.get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale = quantization_utils::get_bias_scale(
                    min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }

            return make_shared<op::QuantizedConvolutionBiasAdd>(input,
                                                                filters,
                                                                mybias,
                                                                sum_input,
                                                                window_movement_strides,
                                                                window_dilation_strides,
                                                                padding_below,
                                                                padding_above,
                                                                data_dilation_strides,
                                                                requantization_scale,
                                                                sum_scale,
                                                                with_relu)
                ->add_provenance_group_members_above({input,
                                                      filters,
                                                      bias,
                                                      sum_input,
                                                      min_input,
                                                      max_input,
                                                      min_filter,
                                                      max_filter,
                                                      min_output,
                                                      max_output,
                                                      min_sum_input,
                                                      max_sum_input});
        }

        shared_ptr<Node>
            QuantizedConvolutionBiasSignedAddBuilder(const Output<Node>& input,
                                                     const Output<Node>& filters,
                                                     const Output<Node>& bias,
                                                     const Output<Node>& sum_input,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const CoordinateDiff& padding_below,
                                                     const CoordinateDiff& padding_above,
                                                     const Strides& data_dilation_strides,
                                                     const Output<Node>& min_input,
                                                     const Output<Node>& max_input,
                                                     const Output<Node>& min_filter,
                                                     const Output<Node>& max_filter,
                                                     const Output<Node>& min_output,
                                                     const Output<Node>& max_output,
                                                     const Output<Node>& min_sum_input,
                                                     const Output<Node>& max_sum_input,
                                                     const bool with_relu)
        {
            auto output_et = with_relu ? element::u8 : element::i8;
            auto input_scale =
                quantization_utils::get_scale(min_input, max_input, input.get_element_type());
            auto filter_scale =
                quantization_utils::get_scale(min_filter, max_filter, filters.get_element_type());
            auto output_scale = quantization_utils::get_scale(min_output, max_output, output_et);
            auto requantization_scale = input_scale * filter_scale / output_scale;

            auto sum_scale = builder::quantization_utils::get_sum_scale(
                min_output, max_output, min_sum_input, max_sum_input);
            if (output_et == element::u8)
            {
                // Need to multiply by two to account for u8 requantization_scale
                auto two = make_constant(element::f32, sum_scale->get_shape(), 2.0f);
                sum_scale = two * sum_scale;
            }

            auto mybias = bias;
            if (bias.get_element_type() != element::i32)
            {
                auto zero = make_constant(element::i32, min_input.get_shape(), 0);
                AxisSet quantization_axes;
                auto bias_scale = quantization_utils::get_bias_scale(
                    min_input, max_input, min_filter, max_filter);
                op::Quantize::RoundMode round_mode =
                    op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                mybias = make_shared<op::Quantize>(
                    bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
            }
            auto qconv = make_shared<op::QuantizedConvolutionBiasSignedAdd>(input,
                                                                            filters,
                                                                            mybias,
                                                                            sum_input,
                                                                            window_movement_strides,
                                                                            window_dilation_strides,
                                                                            padding_below,
                                                                            padding_above,
                                                                            data_dilation_strides,
                                                                            requantization_scale,
                                                                            sum_scale,
                                                                            with_relu);
            return make_shared<op::Convert>(qconv, element::u8)
                ->add_provenance_group_members_above({input,
                                                      filters,
                                                      bias,
                                                      sum_input,
                                                      min_input,
                                                      max_input,
                                                      min_filter,
                                                      max_filter,
                                                      min_output,
                                                      max_output,
                                                      min_sum_input,
                                                      max_sum_input});
        };
    }
}
