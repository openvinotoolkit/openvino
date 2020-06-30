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

#include <cstdint>
#include <memory>
#include <numeric>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "quantize_linear.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector quantize_linear(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    const auto& x = inputs.at(0);
                    std::shared_ptr<ngraph::Node> y_scale = inputs.at(1);
                    std::shared_ptr<ngraph::Node> y_zero_point;
                    if (inputs.size() > 2)
                    {
                        y_zero_point = inputs.at(2);
                    }
                    else
                    {
                        y_zero_point = std::make_shared<default_opset::Constant>(
                            element::u8, Shape{1}, std::uint8_t(0));
                    }

                    const auto& y_zero_point_et = y_zero_point->get_element_type();
                    CHECK_VALID_NODE(
                        node,
                        y_zero_point_et.is_static() &&
                            (y_zero_point_et == element::u8 || y_zero_point_et == element::i8),
                        "\"y_zero_point\" input data type must be static and of 8-bit "
                        "integer type.");

                    const auto& y_scale_et = y_scale->get_element_type();
                    CHECK_VALID_NODE(node,
                                     y_scale_et.is_static(),
                                     "\"y_scale\" input data type must be static.");

                    const auto& x_et = x->get_element_type();

                    if (y_scale_et != x_et)
                    {
                        y_scale = std::make_shared<default_opset::Convert>(y_scale, x_et);
                    }

                    const auto& destination_type = y_zero_point_et;
                    std::shared_ptr<ngraph::Node> output_low;
                    std::shared_ptr<ngraph::Node> output_high;

                    if (destination_type == element::i8)
                    {
                        output_low =
                            std::make_shared<default_opset::Constant>(x_et, Shape{1}, -128);
                        output_high =
                            std::make_shared<default_opset::Constant>(x_et, Shape{1}, 127);
                    }
                    else
                    {
                        output_low = std::make_shared<default_opset::Constant>(x_et, Shape{1}, 0);
                        output_high =
                            std::make_shared<default_opset::Constant>(x_et, Shape{1}, 255);
                    }

                    std::shared_ptr<ngraph::Node> input_low;
                    std::shared_ptr<ngraph::Node> input_high;

                    if (y_scale->is_constant() && y_zero_point->is_constant())
                    {
                        const auto& zero_point =
                            std::make_shared<default_opset::Convert>(y_zero_point, x_et);

                        input_low = std::make_shared<default_opset::Multiply>(
                            y_scale, std::make_shared<default_opset::Add>(output_low, zero_point));
                        input_high = std::make_shared<default_opset::Multiply>(
                            y_scale, std::make_shared<default_opset::Add>(output_high, zero_point));
                    }
                    else
                    {
                        std::shared_ptr<ngraph::Node> reduction_axes;

                        if (x->get_output_partial_shape(0).rank().is_static())
                        {
                            const auto rank = static_cast<size_t>(
                                x->get_output_partial_shape(0).rank().get_length());
                            std::vector<int32_t> axes(rank);
                            std::iota(std::begin(axes), std::end(axes), 0);
                            reduction_axes = std::make_shared<default_opset::Constant>(
                                element::i32, Shape{rank}, axes);
                        }
                        else
                        {
                            const auto& stop = reshape::interpret_as_scalar(
                                std::make_shared<default_opset::ShapeOf>(
                                    std::make_shared<default_opset::ShapeOf>(x, element::i32),
                                    element::i32));
                            reduction_axes = std::make_shared<default_opset::Range>(
                                std::make_shared<default_opset::Constant>(element::i32, Shape{}, 0),
                                stop,
                                std::make_shared<default_opset::Constant>(
                                    element::i32, Shape{}, 1));
                        }

                        input_low = std::make_shared<default_opset::ReduceMin>(x, reduction_axes);
                        input_high = std::make_shared<default_opset::ReduceMax>(x, reduction_axes);
                    }

                    const std::size_t levels = 1 << destination_type.bitwidth();

                    return {std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::FakeQuantize>(
                            x, input_low, input_high, output_low, output_high, levels),
                        destination_type)};
                }
            } // namespace set_1

            namespace set_13
            {
                NodeVector quantize_linear(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    std::shared_ptr<ngraph::Node> x = inputs.at(0);
                    std::shared_ptr<ngraph::Node> y_scale = inputs.at(1);
                    std::shared_ptr<ngraph::Node> y_zero_point = inputs.at(2);

                    // get axis twice with two default values to see if it is set
                    int64_t axis_0{node.get_attribute_value<int64_t>("axis", 0)};
                    int64_t axis_1{node.get_attribute_value<int64_t>("axis", 1)};

                    AxisSet axes;

                    // if axis attribute is set
                    if (axis_0 == axis_1)
                    {
                        // positive axis
                        if (axis_0 >= 0)
                        {
                            axes.insert(axis_0);
                        }
                        // negative axis
                        else if (axis_0 < 0)
                        {
                            axes.insert(x->get_shape().size() + axis_0);
                        }
                    }

                    Shape y_scale_shape = y_scale->get_shape();
                    Shape y_zero_point_shape = y_zero_point->get_shape();

                    return {std::make_shared<ngraph::opset0::Quantize>(
                        x,
                        y_scale,
                        y_zero_point,
                        y_zero_point->get_element_type(),
                        axes,
                        ngraph::opset0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)};
                }

            } // namespace set_13

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
