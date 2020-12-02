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
#include <map>
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                static const std::unordered_set<std::string> supported_modes = {
                    "nearest", "linear", "cubic"};

                static const std::unordered_set<std::string> supported_transforms = {
                    "half_pixel",
                    "pytorch_half_pixel",
                    "align_corners",
                    "asymmetric",
                    "tf_half_pixel_for_nn"};

                using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;
                static const std::map<std::string, int> interp_mode_map = {
                    {"nearest", static_cast<int>(InterpolateMode::nearest)},
                    {"linear", static_cast<int>(InterpolateMode::linear_onnx)},
                    {"cubic", static_cast<int>(InterpolateMode::cubic)}};

                using Transform_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
                static const std::map<std::string, int> transform_mode_map = {
                    {"half_pixel", static_cast<int>(Transform_mode::half_pixel)},
                    {"pytorch_half_pixel", static_cast<int>(Transform_mode::pytorch_half_pixel)},
                    {"align_corners", static_cast<int>(Transform_mode::align_corners)},
                    {"asymmetric", static_cast<int>(Transform_mode::asymmetric)},
                    {"tf_half_pixel_for_nn",
                     static_cast<int>(Transform_mode::tf_half_pixel_for_nn)}};

                using Nearest_mode = ngraph::op::v4::Interpolate::NearestMode;
                static const std::map<std::string, int> nearest_mode_map = {
                    {"round_prefer_floor", static_cast<int>(Nearest_mode::round_prefer_floor)},
                    {"round_prefer_ceil", static_cast<int>(Nearest_mode::round_prefer_ceil)},
                    {"floor", static_cast<int>(Nearest_mode::floor)},
                    {"ceil", static_cast<int>(Nearest_mode::ceil)}};

                static bool is_supported_str_value(const std::unordered_set<std::string>& modes,
                                                   const std::string& checked_mode)
                {
                    return std::find(modes.begin(), modes.end(), checked_mode) != modes.end();
                }

                static int mode_as_int(const std::map<std::string, int>& converting_map,
                                       const std::string& mode)
                {
                    int result = 0;
                    auto it = converting_map.find(mode);
                    if (it != converting_map.end())
                    {
                        result = it->second;
                    }
                    return result;
                }

                using InterpolateV4Attrs = ngraph::op::v4::Interpolate::InterpolateAttrs;

                InterpolateV4Attrs get_resize_attrs(const onnx_import::Node& node)
                {
                    auto get_str_attr = [&node](const std::string& name,
                                                const std::string& default_value) {
                        return node.get_attribute_value<std::string>(name, default_value);
                    };

                    const auto mode = get_str_attr("mode", "nearest");
                    const auto transform_mode =
                        get_str_attr("coordinate_transformation_mode", "half_pixel");
                    const auto nearest_mode = get_str_attr("nearest_mode", "round_prefer_floor");

                    if (!is_supported_str_value(supported_modes, mode))
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_modes)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        supported_modes_str.pop_back();
                        supported_modes_str.pop_back();
                        CHECK_VALID_NODE(node,
                                         false,
                                         mode,
                                         " - this type of interpolation mode is not supported."
                                         " Choose one of the following modes: ",
                                         supported_modes_str);
                    }

                    if (!is_supported_str_value(supported_transforms, transform_mode))
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_transforms)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        supported_modes_str.pop_back();
                        supported_modes_str.pop_back();
                        CHECK_VALID_NODE(node,
                                         false,
                                         transform_mode,
                                         " - this type of coordinate transformation mode is not "
                                         "supported. Choose one of the following modes: ",
                                         supported_modes_str);
                    }

                    InterpolateV4Attrs attrs;
                    attrs.mode = static_cast<InterpolateMode>(mode_as_int(interp_mode_map, mode));
                    attrs.coordinate_transformation_mode = static_cast<Transform_mode>(
                        mode_as_int(transform_mode_map, transform_mode));
                    attrs.nearest_mode =
                        static_cast<Nearest_mode>(mode_as_int(nearest_mode_map, nearest_mode));
                    attrs.antialias = false;
                    attrs.cube_coeff = node.get_attribute_value<float>("cubic_coeff_a", -0.75);

                    auto zero_pad = std::vector<size_t>(1, 0);

                    attrs.pads_begin = zero_pad;
                    attrs.pads_end = zero_pad;

                    return attrs;
                }

                std::shared_ptr<ngraph::Node>
                    calculate_output_shape_based_on_scales(const Output<ngraph::Node>& data,
                                                           const Output<ngraph::Node>& scales)
                {
                    const auto& data_shape = data.get_partial_shape();
                    const auto& scales_shape = scales.get_partial_shape();

                    if (ngraph::op::is_constant(scales.get_node()) && data_shape.is_static())
                    {
                        const auto scales_const =
                            as_type_ptr<default_opset::Constant>(scales.get_node_shared_ptr());

                        const auto scales_vector = scales_const->cast_vector<float>();
                        const auto data_static_shape = data_shape.to_shape();

                        std::vector<int64_t> output_shape;
                        for (size_t i = 0; i < data_static_shape.size(); ++i)
                        {
                            output_shape.push_back(
                                std::floor(data_static_shape.at(i) * scales_vector.at(i)));
                        }
                        auto output_shape_const = default_opset::Constant::create(
                            element::Type_t::u64, Shape({output_shape.size()}), output_shape);

                        return output_shape_const;
                    }

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), scales.get_element_type());
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    const auto output_shape = std::make_shared<default_opset::Convert>(
                        multiply, ngraph::element::Type_t::i64);

                    return output_shape;
                }

                std::shared_ptr<ngraph::Node>
                    calculate_scales_based_on_sizes(const Output<ngraph::Node>& data,
                                                    const Output<ngraph::Node>& sizes)
                {
                    const auto& data_shape = data.get_partial_shape();
                    const auto& sizes_shape = sizes.get_partial_shape();

                    const float epsilon = 1.0e-5;

                    if (ngraph::op::is_constant(sizes.get_node()) && data_shape.is_static())
                    {
                        const auto sizes_const =
                            as_type_ptr<default_opset::Constant>(sizes.get_node_shared_ptr());

                        const auto sizes_vector = sizes_const->cast_vector<int64_t>();
                        const auto data_static_shape = data_shape.to_shape();

                        std::vector<float> scales;
                        for (size_t i = 0; i < data_static_shape.size(); ++i)
                        {
                            float scale = static_cast<float>(sizes_vector.at(i)) /
                                              static_cast<float>(data_static_shape.at(i)) +
                                          epsilon;
                            scales.push_back(scale);
                        }
                        auto scales_const = default_opset::Constant::create(
                            element::Type_t::f32, Shape({scales.size()}), scales);

                        return scales_const;
                    }

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data),
                        ngraph::element::Type_t::f32);
                    const auto converted_sizes = std::make_shared<default_opset::Convert>(
                        sizes, ngraph::element::Type_t::f32);
                    const auto divide =
                        std::make_shared<default_opset::Divide>(converted_sizes, shape_of_data);
                    const auto eps_node = std::make_shared<default_opset::Constant>(
                        ngraph::element::Type_t::f32, Shape{}, epsilon);
                    const auto scales = std::make_shared<default_opset::Add>(divide, eps_node);

                    return scales;
                }

                OutputVector build_resize(const Node& node,
                                          const std::shared_ptr<ngraph::Node>& output_shape,
                                          const AxisSet& axes)
                {
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

                    auto attrs = ngraph::op::v0::InterpolateAttrs();
                    attrs.axes = axes;
                    attrs.mode = mode;
                    attrs.align_corners = false;

                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);

                    return {
                        std::make_shared<ngraph::op::v0::Interpolate>(data, output_shape, attrs)};
                }
            } // namespace

            namespace set_11
            {
                OutputVector resize(const onnx_import::Node& node)
                {
                    // roi input (inputs.at(2)) is ignored because it is used only
                    // in "tf_crop_and_resize" which is not handled now
                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);
                    const auto& data_shape = data.get_partial_shape();

                    auto attrs = get_resize_attrs(node);

                    if (inputs.size() == 4) // sizes input is provided
                    {
                        attrs.shape_calculation_mode =
                            default_opset::Interpolate::ShapeCalcMode::sizes;
                        const auto& sizes = inputs.at(3);
                        const auto& sizes_shape = sizes.get_partial_shape();

                        CHECK_VALID_NODE(
                            node,
                            (sizes_shape.is_static() || data_shape.rank().is_static()),
                            " Data rank or shape of sizes input is required to be static.");

                        const auto scales = calculate_scales_based_on_sizes(data, sizes);

                        return {std::make_shared<default_opset::Interpolate>(
                            data, sizes, scales, attrs)};
                    }

                    attrs.shape_calculation_mode =
                        default_opset::Interpolate::ShapeCalcMode::scales;

                    const auto& scales = inputs.at(2);
                    const auto& scales_shape = scales.get_partial_shape();

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of scales input is required to be static.");

                    const auto output_shape = calculate_output_shape_based_on_scales(data, scales);
                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales, attrs)};
                }
            } // namespace set_11

            namespace set_1
            {
                OutputVector resize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);
                    const auto& scales = inputs.at(1);

                    const auto& data_shape = data.get_partial_shape();
                    const auto& scales_shape = scales.get_partial_shape();

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of scales input is required to be static.");

                    size_t axes_size = scales_shape.is_static() ? scales_shape[0].get_length()
                                                                : data_shape.rank().get_length();

                    const auto output_shape = calculate_output_shape_based_on_scales(data, scales);
                    return build_resize(
                        node, output_shape, AxisSet(common::get_monotonic_range(axes_size)));
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
