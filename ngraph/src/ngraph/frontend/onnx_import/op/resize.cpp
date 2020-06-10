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

#include "resize.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector resize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);
                    // const auto scales = inputs.at(1);

                    const auto data_shape = data->get_output_partial_shape(0);
                    // const auto scales_shape = scales->get_output_partial_shape(0);

                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    const auto scales = node.get_attribute_value<std::vector<float>>("scales", {});

                    std::unordered_set<std::string> supported_modes = {"nearest", "linear"};
                    bool is_mode_supported =
                        (std::find(supported_modes.begin(), supported_modes.end(), mode) !=
                         supported_modes.end());

                    if (!is_mode_supported)
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_modes)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        CHECK_VALID_NODE(node,
                                         is_mode_supported,
                                         mode,
                                         " - this type of interpolation mode is not supported."
                                         " Choose one of the following modes: ",
                                         supported_modes_str);
                    }

                    size_t axes_size = scales.size();

                    AxisSet axes;
                    for (int ax = 0; ax < axes_size; ++ax)
                    {
                        axes.insert(ax);
                    }

                    auto attrs = ngraph::op::v0::InterpolateAttrs();
                    attrs.axes = axes;
                    attrs.mode = mode;
                    attrs.align_corners = false;

                    const auto scales_const = default_opset::Constant::create(
                        ngraph::element::f32, Shape({scales.size()}), scales);
                    if (data_shape.is_static())
                    {
                        auto data_static_shape = data_shape.to_shape();

                        std::vector<int64_t> output_shape;
                        for (size_t i = 0; i < data_static_shape.size(); ++i)
                        {
                            output_shape.push_back(
                                std::floor(data_static_shape.at(i) * scales.at(i)));
                        }
                        auto output_shape_const = default_opset::Constant::create(
                            element::u64, Shape({output_shape.size()}), output_shape);

                        return {std::make_shared<default_opset::Interpolate>(
                            data, output_shape_const, attrs)};
                    }

                    auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales_const);
                    auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);
                    return {
                        std::make_shared<default_opset::Interpolate>(data, output_shape, attrs)};
                }

            } // namespace set_1

            namespace set_9
            {
                NodeVector resize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);
                    const auto scales = inputs.at(1);

                    const auto data_shape = data->get_output_partial_shape(0);
                    const auto scales_shape = scales->get_output_partial_shape(0);

                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");

                    std::unordered_set<std::string> supported_modes = {"nearest", "linear"};
                    bool is_mode_supported =
                        (std::find(supported_modes.begin(), supported_modes.end(), mode) !=
                         supported_modes.end());

                    if (!is_mode_supported)
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_modes)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        CHECK_VALID_NODE(node,
                                         is_mode_supported,
                                         mode,
                                         " - this type of interpolation mode is not supported."
                                         " Choose one of the following modes: ",
                                         supported_modes_str);
                    }

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of Scales input is required to be static.");

                    size_t axes_size = scales_shape.is_static() ? scales_shape.to_shape().at(0)
                                                                : data_shape.rank().get_length();
                    AxisSet axes;
                    for (int ax = 0; ax < axes_size; ++ax)
                    {
                        axes.insert(ax);
                    }

                    auto attrs = ngraph::op::v0::InterpolateAttrs();
                    attrs.axes = axes;
                    attrs.mode = mode;
                    attrs.align_corners = false;

                    if (scales->is_constant() && data_shape.is_static())
                    {
                        const auto scales_const =
                            as_type_ptr<default_opset::Constant>(scales->shared_from_this());

                        auto scales_vector = scales_const->cast_vector<float>();
                        auto data_static_shape = data_shape.to_shape();

                        std::vector<int64_t> output_shape;
                        for (size_t i = 0; i < data_static_shape.size(); ++i)
                        {
                            output_shape.push_back(
                                std::floor(data_static_shape.at(i) * scales_vector.at(i)));
                        }
                        auto output_shape_const = default_opset::Constant::create(
                            element::u64, Shape({output_shape.size()}), output_shape);

                        return {std::make_shared<default_opset::Interpolate>(
                            data, output_shape_const, attrs)};
                    }

                    auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);
                    return {
                        std::make_shared<default_opset::Interpolate>(data, output_shape, attrs)};
                }

            } // namespace set_9

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
