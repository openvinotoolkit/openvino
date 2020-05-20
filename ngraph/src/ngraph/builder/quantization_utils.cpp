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

#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace quantization_utils
        {
            std::shared_ptr<Node> max_abs(const Output<Node>& a, const Output<Node>& b)
            {
                auto abs_a = std::make_shared<op::Abs>(a);
                auto abs_b = std::make_shared<op::Abs>(b);
                return std::make_shared<op::Maximum>(abs_a, abs_b)
                    ->add_provenance_group_members_above({a, b});
            }

            std::shared_ptr<Node> get_scale(const Output<Node>& input_min_range,
                                            const Output<Node>& input_max_range,
                                            const ngraph::element::Type& quant_type,
                                            bool bump_by_eps)
            {
                auto type = input_min_range.get_element_type();
                if (type != input_max_range.get_element_type())
                {
                    throw ngraph_error("get_scale: min and max must have same type");
                }

                auto shape = input_min_range.get_shape();
                if (shape != input_max_range.get_shape())
                {
                    throw ngraph_error("get_scale: min and max must have same shape");
                }

                auto min_range = input_min_range;
                auto max_range = input_max_range;

                if (bump_by_eps)
                {
                    auto zero = make_constant(type, shape, 0);
                    min_range = std::make_shared<op::Minimum>(zero, input_min_range);

                    auto max_abs_input_range = max_abs(input_min_range, input_max_range);

                    auto one = make_constant(type, shape, 1);
                    auto hundred = make_constant(type, shape, 100);
                    auto epsilon =
                        std::make_shared<op::Maximum>(one, max_abs_input_range) / hundred;

                    max_range = std::make_shared<op::Maximum>(input_max_range, min_range + epsilon);
                    max_range = std::make_shared<op::Maximum>(zero, max_range);
                }

                size_t bw = quant_type.bitwidth();
                float range = static_cast<float>(
                    (quant_type.is_signed() ? std::pow(2, (bw - 1)) : std::pow(2, bw)) - 1);

                auto max_abs_range = max_abs(min_range, max_range);
                auto target_range = make_constant(type, shape, range);

                return (max_abs_range / target_range)
                    ->add_provenance_group_members_above({input_min_range, input_max_range});
            }

            std::shared_ptr<Node> get_bias_scale(Output<Node> min_input,
                                                 Output<Node> max_input,
                                                 Output<Node> min_filter,
                                                 Output<Node> max_filter)
            {
                auto type = min_input.get_element_type();
                if (type != max_input.get_element_type() || type != min_filter.get_element_type() ||
                    type != max_filter.get_element_type())
                {
                    throw ngraph_error("get_bias_scale: min and max must have same type");
                }

                auto shape = min_input.get_shape();
                if (shape != max_input.get_shape() || shape != min_filter.get_shape() ||
                    shape != max_filter.get_shape())
                {
                    throw ngraph_error("get_bias_scale: min and max must have same shape");
                }

                auto max_abs_input_range = max_abs(min_input, max_input);
                auto max_abs_filter_range = max_abs(min_filter, max_filter);
                auto range = make_constant(type,
                                           shape,
                                           std::numeric_limits<uint8_t>::max() *
                                               std::numeric_limits<int8_t>::max());

                // Inverting the scale calculation here as the Quantize op passes scale as 1/scale.
                return (max_abs_input_range * max_abs_filter_range) / range;
            }

            std::shared_ptr<Node> get_sum_scale(Output<Node> min_freezed_output_conv_1,
                                                Output<Node> max_freezed_output_conv_1,
                                                Output<Node> min_freezed_output_conv_2,
                                                Output<Node> max_freezed_output_conv_2)
            {
                auto type = min_freezed_output_conv_1.get_element_type();
                if (type != max_freezed_output_conv_1.get_element_type() ||
                    type != min_freezed_output_conv_2.get_element_type() ||
                    type != max_freezed_output_conv_2.get_element_type())
                {
                    throw ngraph_error("get_sum_scale: min and max must have same type");
                }

                auto shape = min_freezed_output_conv_1.get_shape();
                if (shape != max_freezed_output_conv_1.get_shape() ||
                    shape != min_freezed_output_conv_2.get_shape() ||
                    shape != max_freezed_output_conv_2.get_shape())
                {
                    throw ngraph_error("get_sum_scale: min and max must have same shape");
                }

                auto max_abs_conv_1 = max_abs(min_freezed_output_conv_1, max_freezed_output_conv_1);
                auto max_abs_conv_2 = max_abs(min_freezed_output_conv_2, max_freezed_output_conv_2);
                return max_abs_conv_2 / max_abs_conv_1;
            }

            std::shared_ptr<Node> get_dot_scale(Output<Node> min_input,
                                                Output<Node> max_input,
                                                Output<Node> min_filter,
                                                Output<Node> max_filter,
                                                Output<Node> min_freezed_output,
                                                Output<Node> max_freezed_output,
                                                const ngraph::element::Type& input_type,
                                                const ngraph::element::Type& output_type,
                                                const bool requantize)
            {
                auto type = min_input.get_element_type();
                if (type != max_input.get_element_type() || type != min_filter.get_element_type() ||
                    type != max_filter.get_element_type() ||
                    type != min_freezed_output.get_element_type() ||
                    type != max_freezed_output.get_element_type())
                {
                    throw ngraph_error("get_dot_scale: min and max must have same type");
                }

                auto shape = min_input.get_shape();
                if (shape != max_input.get_shape() || shape != min_filter.get_shape() ||
                    shape != max_filter.get_shape() || shape != min_freezed_output.get_shape() ||
                    shape != max_freezed_output.get_shape())
                {
                    throw ngraph_error("get_dot_scale: min and max must have same shape");
                }
                auto data_scale = get_scale(min_input, max_input, input_type);
                auto weight_scale = get_scale(min_filter, max_filter, element::i8);
                auto out_scale = get_scale(min_freezed_output, max_freezed_output, output_type);
                if (requantize)
                {
                    return data_scale * weight_scale / out_scale;
                }
                else
                {
                    return data_scale * weight_scale;
                }
            }

            void
                check_concat(const NodeVector& args, const NodeVector& mins, const NodeVector& maxs)
            {
                auto size = args.size();
                if (size != mins.size() || size != maxs.size())
                {
                    throw ngraph_error("Min and Max node vectors must be of same length");
                }
                for (size_t i = 0; i < size; i++)
                {
                    auto min = mins[i];
                    auto max = maxs[i];
                    auto type = min->get_element_type();
                    if (type != max->get_element_type())
                    {
                        throw ngraph_error("check_concat: min and max must have same type");
                    }

                    if (min->get_shape() != Shape{1} || max->get_shape() != Shape{1})
                    {
                        throw ngraph_error("check_concat: min/max shape not Shape{1}: " +
                                           vector_to_string(min->get_shape()) +
                                           vector_to_string(max->get_shape()));
                    }
                }
            }
        }
    }
}
