// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/validation_util.hpp"
#include "op/quantize_linear.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace detail
            {
                namespace
                {
                    Output<ngraph::Node> get_zero_point(const OutputVector& inputs)
                    {
                        if (inputs.size() > 2)
                        {
                            return inputs.at(2);
                        }
                        else
                        {
                            return std::make_shared<default_opset::Constant>(
                                element::u8, Shape{1}, std::uint8_t(0));
                        }
                    }

                    void validate_zero_point_type(const Node& onnx_node,
                                                  const Output<ngraph::Node>& y_zero_point)
                    {
                        const auto& y_zero_point_et = y_zero_point.get_element_type();
                        CHECK_VALID_NODE(
                            onnx_node,
                            y_zero_point_et.is_static() &&
                                (y_zero_point_et == element::u8 || y_zero_point_et == element::i8),
                            "\"y_zero_point\" input data type must be static and of 8-bit "
                            "integer type.");
                    }

                    Output<ngraph::Node> validate_scale(const Node& onnx_node,
                                                        const Output<ngraph::Node>& y_scale)
                    {
                        const auto& y_scale_et = y_scale.get_element_type();
                        CHECK_VALID_NODE(onnx_node,
                                         y_scale_et.is_static(),
                                         "\"y_scale\" input data type must be static.");
                        if (y_scale_et != element::f32)
                        {
                            return std::make_shared<default_opset::Convert>(y_scale, element::f32);
                        }
                        return y_scale;
                    }

                    Output<ngraph::Node> validate_data(const Node& onnx_node,
                                                       const Output<ngraph::Node>& data)
                    {
                        const auto& data_et = data.get_element_type();
                        CHECK_VALID_NODE(onnx_node,
                                         data_et.is_static(),
                                         "\"x\" input data type must be static.");

                        if (data_et != element::f32)
                        {
                            return std::make_shared<default_opset::Convert>(data, element::f32);
                        }
                        return data;
                    }

                    std::tuple<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>
                        get_output_bands(const element::Type& destination_type,
                                         const element::Type& data_type)
                    {
                        std::shared_ptr<ngraph::Node> output_low;
                        std::shared_ptr<ngraph::Node> output_high;

                        if (destination_type == element::i8)
                        {
                            output_low = std::make_shared<default_opset::Constant>(
                                data_type, Shape{1}, -128);
                            output_high =
                                std::make_shared<default_opset::Constant>(data_type, Shape{1}, 127);
                        }
                        else
                        {
                            output_low =
                                std::make_shared<default_opset::Constant>(data_type, Shape{1}, 0);
                            output_high =
                                std::make_shared<default_opset::Constant>(data_type, Shape{1}, 255);
                        }

                        return std::make_tuple(output_low, output_high);
                    }

                    std::tuple<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>
                        get_input_bands(const Output<ngraph::Node>& y_scale,
                                        const Output<ngraph::Node>& y_zero_point,
                                        const std::shared_ptr<ngraph::Node>& output_low,
                                        const std::shared_ptr<ngraph::Node>& output_high,
                                        const element::Type& data_type)
                    {
                        std::shared_ptr<ngraph::Node> input_low;
                        std::shared_ptr<ngraph::Node> input_high;
                        const auto& zero_point =
                            std::make_shared<default_opset::Convert>(y_zero_point, data_type);

                        input_low = std::make_shared<default_opset::Multiply>(
                            y_scale,
                            std::make_shared<default_opset::Subtract>(output_low, zero_point));
                        input_high = std::make_shared<default_opset::Multiply>(
                            y_scale,
                            std::make_shared<default_opset::Subtract>(output_high, zero_point));

                        return std::make_tuple(input_low, input_high);
                    }

                    std::shared_ptr<ngraph::Node>
                        make_fake_quantize(const Output<ngraph::Node>& y_scale,
                                           const Output<ngraph::Node>& y_zero_point,
                                           const Output<ngraph::Node>& data)
                    {
                        const element::Type& destination_type = y_zero_point.get_element_type();
                        const element::Type& data_type = data.get_element_type();

                        std::shared_ptr<ngraph::Node> output_low;
                        std::shared_ptr<ngraph::Node> output_high;
                        std::tie(output_low, output_high) =
                            detail::get_output_bands(destination_type, data_type);

                        std::shared_ptr<ngraph::Node> input_low;
                        std::shared_ptr<ngraph::Node> input_high;
                        std::tie(input_low, input_high) = detail::get_input_bands(
                            y_scale, y_zero_point, output_low, output_high, data_type);

                        const std::size_t levels = 1 << destination_type.bitwidth();

                        return std::make_shared<default_opset::Convert>(
                            std::make_shared<default_opset::FakeQuantize>(
                                data, input_low, input_high, output_low, output_high, levels),
                            destination_type);
                    }
                } // namespace
            }     // namespace detail

            namespace set_1
            {
                OutputVector quantize_linear(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    auto x = inputs.at(0);
                    auto y_scale = inputs.at(1);
                    auto y_zero_point = detail::get_zero_point(inputs);

                    x = detail::validate_data(node, x);
                    detail::validate_zero_point_type(node, y_zero_point);
                    y_scale = detail::validate_scale(node, y_scale);

                    return {detail::make_fake_quantize(y_scale, y_zero_point, x)};
                }
            } // namespace set_1

            namespace set_13
            {
                OutputVector quantize_linear(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    auto x = inputs.at(0);
                    auto y_scale = inputs.at(1);
                    auto y_zero_point = detail::get_zero_point(inputs);

                    x = detail::validate_data(node, x);
                    detail::validate_zero_point_type(node, y_zero_point);
                    y_scale = detail::validate_scale(node, y_scale);

                    const auto& x_shape = x.get_partial_shape();

                    int64_t axis{node.get_attribute_value<int64_t>("axis", 1)};
                    axis = normalize_axis(node.get_description(), axis, x_shape.rank());

                    const auto& y_scale_shape = y_scale.get_partial_shape();
                    const auto& y_zero_point_shape = y_zero_point.get_partial_shape();

                    if (y_scale_shape.rank().is_static() &&
                        y_scale_shape.rank().get_length() == 1 && x_shape.rank().is_static() &&
                        x_shape[axis].is_static())
                    {
                        CHECK_VALID_NODE(
                            node,
                            y_scale_shape[0].same_scheme(x_shape[axis]),
                            "The number of quantization scale elements ",
                            y_scale_shape[0],
                            " must match the number of respective input data axis size: ",
                            x_shape[axis]);

                        Shape target_shape(x_shape.rank().get_length(), 1);
                        target_shape[axis] = static_cast<size_t>(x_shape[axis].get_length());

                        y_scale = builder::opset1::reshape(y_scale, target_shape);
                    }

                    if (y_zero_point_shape.rank().is_static() &&
                        y_zero_point_shape.rank().get_length() == 1 && x_shape.rank().is_static() &&
                        x_shape[axis].is_static())
                    {
                        CHECK_VALID_NODE(
                            node,
                            y_zero_point_shape[0].same_scheme(x_shape[axis]),
                            "The number of quantization zero point elements ",
                            y_zero_point_shape[0],
                            " must match the number of respective input data axis size: ",
                            x_shape[axis]);

                        Shape target_shape(x_shape.rank().get_length(), 1);
                        target_shape[axis] = static_cast<size_t>(x_shape[axis].get_length());

                        y_zero_point = builder::opset1::reshape(y_zero_point, target_shape);
                    }

                    return {detail::make_fake_quantize(y_scale, y_zero_point, x)};
                }

            } // namespace set_13

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
