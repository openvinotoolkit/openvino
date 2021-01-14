//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/op/upsample.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                bool check_mode_support(const onnx_import::Node& node, const std::string& mode)
                {
                    const std::unordered_set<std::string> supported_modes = {"nearest", "linear"};
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
                    return is_mode_supported;
                }

                using Transform_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
                using Nearest_mode = ngraph::op::v4::Interpolate::NearestMode;
                using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;
                using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;

                InterpolateMode convert_mode(const std::string& mode_str)
                {
                    InterpolateMode result = InterpolateMode::nearest;
                    if (mode_str == "linear")
                    {
                        result = InterpolateMode::linear_onnx;
                    }
                    return result;
                }
            }

            namespace set_1
            {
                OutputVector upsample(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);

                    const auto data_shape = data.get_partial_shape();

                    const auto scales = node.get_attribute_value<std::vector<float>>("scales");
                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    check_mode_support(node, mode);

                    auto attrs = ngraph::op::v4::Interpolate::InterpolateAttrs();
                    attrs.mode = convert_mode(mode);
                    attrs.shape_calculation_mode = ShapeCalcMode::scales;
                    attrs.coordinate_transformation_mode = Transform_mode::half_pixel;
                    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
                    attrs.antialias = false;
                    attrs.cube_coeff = -0.75;

                    if (attrs.mode == InterpolateMode::linear_onnx)
                    {
                        attrs.coordinate_transformation_mode = Transform_mode::asymmetric;
                    }

                    auto zero_pad = std::vector<size_t>(1, 0);

                    attrs.pads_begin = zero_pad;
                    attrs.pads_end = zero_pad;

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

                        const auto scales_const = default_opset::Constant::create(
                            ngraph::element::f32, Shape({scales.size()}), scales);

                        return {std::make_shared<default_opset::Interpolate>(
                            data, output_shape_const, scales_const, attrs)};
                    }

                    const auto scales_const = default_opset::Constant::create(
                        ngraph::element::f32, Shape({scales.size()}), scales);

                    auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales_const);
                    auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);

                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales_const, attrs)};
                }

            } // namespace set_1

            namespace set_9
            {
                OutputVector upsample(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);
                    const auto scales = inputs.at(1);

                    const auto data_shape = data.get_partial_shape();
                    const auto scales_shape = scales.get_partial_shape();

                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    check_mode_support(node, mode);

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of Scales input is required to be static.");

                    auto attrs = ngraph::op::v4::Interpolate::InterpolateAttrs();
                    attrs.mode = convert_mode(mode);
                    attrs.shape_calculation_mode = ShapeCalcMode::scales;
                    attrs.coordinate_transformation_mode = Transform_mode::half_pixel;
                    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
                    attrs.antialias = false;
                    attrs.cube_coeff = -0.75;

                    if (attrs.mode == InterpolateMode::linear_onnx)
                    {
                        attrs.coordinate_transformation_mode = Transform_mode::asymmetric;
                    }

                    auto zero_pad = std::vector<size_t>(1, 0);

                    attrs.pads_begin = zero_pad;
                    attrs.pads_end = zero_pad;

                    if (ngraph::op::is_constant(scales.get_node()) && data_shape.is_static())
                    {
                        const auto scales_const =
                            as_type_ptr<default_opset::Constant>(scales.get_node_shared_ptr());

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
                            data, output_shape_const, scales, attrs)};
                    }

                    auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);

                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales, attrs)};
                }

            } // namespace set_9

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
