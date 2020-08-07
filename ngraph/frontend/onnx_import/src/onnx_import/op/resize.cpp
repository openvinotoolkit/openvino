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
                            element::u64, Shape({output_shape.size()}), output_shape);

                        return output_shape_const;
                    }

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), scales.get_element_type());
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    const auto output_shape =
                        std::make_shared<default_opset::Convert>(multiply, ngraph::element::i64);

                    return output_shape;
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
                        std::make_shared<default_opset::Interpolate>(data, output_shape, attrs)};
                }
            }

            namespace set_11
            {
                OutputVector resize(const onnx_import::Node& node)
                {
                    // cubic_coeff_a, extrapolation_value attributes are ignored
                    // (they do not have influence on supported modes)
                    const auto coordinate_transformation_mode =
                        node.get_attribute_value<std::string>("coordinate_transformation_mode",
                                                              "half_pixel");
                    CHECK_VALID_NODE(
                        node,
                        coordinate_transformation_mode == "asymmetric",
                        coordinate_transformation_mode,
                        " - this type of coordinate transformation mode is not supported."
                        " Only asymmetric mode is supported by current version.");

                    const auto exclude_outside =
                        node.get_attribute_value<int64_t>("exclude_outside ", 0);
                    CHECK_VALID_NODE(node,
                                     exclude_outside == 0,
                                     "Expected exclude_outside=",
                                     exclude_outside,
                                     " mode is not supported. ",
                                     "Current version supports only exclude_outside equals `0`.");

                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    if (mode == "nearest")
                    {
                        const auto nearest_mode = node.get_attribute_value<std::string>(
                            "nearest_mode", "round_prefer_floor");
                        CHECK_VALID_NODE(
                            node,
                            nearest_mode == "floor",
                            "Expected nearest_mode=",
                            nearest_mode,
                            " mode is not supported. ",
                            "Current version support only nearest_mode equals `floor`");
                    }

                    // roi input (inputs.at(2)) is ignored because it is used only
                    // in "tf_crop_and_resize" which is not handled now
                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);
                    const auto& data_shape = data.get_partial_shape();

                    if (inputs.size() == 4) // sizes input is provided
                    {
                        const auto& sizes = inputs.at(3);
                        const auto& sizes_shape = sizes.get_partial_shape();

                        CHECK_VALID_NODE(
                            node,
                            (sizes_shape.is_static() || data_shape.rank().is_static()),
                            " Data rank or shape of sizes input is required to be static.");

                        size_t axes_size = sizes_shape.is_static() ? sizes_shape[0].get_length()
                                                                   : data_shape.rank().get_length();

                        return build_resize(node,
                                            sizes.get_node_shared_ptr(),
                                            AxisSet(common::get_monotonic_range(axes_size)));
                    }

                    const auto& scales = inputs.at(2);
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
