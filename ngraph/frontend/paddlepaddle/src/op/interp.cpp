// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interp.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                std::shared_ptr<ngraph::Node>
                    calculate_output_shape_based_on_scales(const Output<ngraph::Node>& data,
                                                           const std::vector<float>& scale,
                                                           Output<ngraph::Node>& scales)
                {
                    FRONT_END_GENERAL_CHECK(scale.size() > 0);
                    if (scale.size() == 1)
                        scales = opset6::Constant::create<float>(
                            element::f32, Shape{4}, {1, 1, scale[0], scale[0]});
                    else if (scale.size() == 2)
                        scales = opset6::Constant::create<float>(
                            element::f32, Shape{4}, {1, 1, scale[0], scale[1]});
                    else if (scale.size() == 3)
                        scales = opset6::Constant::create<float>(
                            element::f32, Shape{4}, {1, scale[0], scale[1], scale[2]});
                    else
                        scales = opset6::Constant::create<float>(
                            element::f32,
                            Shape{scale.size()},
                            std::vector<float>(scale.begin(), scale.end()));
                    const auto shape_of_data = std::make_shared<opset6::Convert>(
                        std::make_shared<opset6::ShapeOf>(data), scales.get_element_type());
                    const auto multiply = std::make_shared<opset6::Multiply>(shape_of_data, scales);
                    const auto output_shape =
                        std::make_shared<opset6::Convert>(multiply, ngraph::element::i64);

                    return output_shape;
                }

                std::shared_ptr<ngraph::Node>
                    calculate_scales_based_on_sizes(const Output<ngraph::Node>& data,
                                                    const Output<ngraph::Node>& sizes)
                {
                    const float epsilon = 1.0e-5;
                    const auto shape_of_data = std::make_shared<opset6::Convert>(
                        std::make_shared<opset6::ShapeOf>(data), ngraph::element::f32);
                    const auto converted_sizes =
                        std::make_shared<opset6::Convert>(sizes, ngraph::element::f32);
                    const auto divide =
                        std::make_shared<opset6::Divide>(converted_sizes, shape_of_data);
                    const auto eps_node =
                        std::make_shared<opset6::Constant>(ngraph::element::f32, Shape{}, epsilon);
                    const auto scales = std::make_shared<opset6::Add>(divide, eps_node);

                    return scales;
                }

                std::shared_ptr<ngraph::Node>
                    extract_out_sizes(const Output<ngraph::Node>& data,
                                      const std::vector<int64_t>& out_sizes)
                {
                    const auto shape_of_x = std::make_shared<opset6::ShapeOf>(data);
                    auto shape_begin = opset6::Constant::create(element::i64, {1}, {0});
                    auto shape_end = opset6::Constant::create(element::i64, Shape{1}, {2});
                    auto nc_node = std::make_shared<opset6::StridedSlice>(shape_of_x,
                                                                          shape_begin,
                                                                          shape_end,
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0});
                    auto hw_node =
                        opset6::Constant::create<int64_t>(element::i64, Shape{2}, out_sizes);
                    return std::make_shared<opset6::Concat>(OutputVector{nc_node, hw_node}, 0);
                }

                // TODO support different data_layout #55170

                NamedOutputs interpolate(const NodeContext& node,
                                         const ngraph::opset6::Interpolate::InterpolateMode& mode)
                {
                    auto x = node.get_ng_input("X");
                    using InterpolateMode = ngraph::opset6::Interpolate::InterpolateMode;
                    using CoordinateTransformMode =
                        ngraph::opset6::Interpolate::CoordinateTransformMode;
                    using Nearest_mode = ngraph::opset6::Interpolate::NearestMode;
                    using InterpolateAttrs = ngraph::opset6::Interpolate::InterpolateAttrs;
                    using ShapeCalcMode = ngraph::opset6::Interpolate::ShapeCalcMode;

                    InterpolateAttrs attrs;

                    attrs.mode = mode;

                    auto out_w = node.get_attribute<int>("out_w");
                    auto out_h = node.get_attribute<int>("out_h");
                    auto scale = node.get_attribute<std::vector<float>>("scale");
                    Output<Node> scales;
                    Output<Node> target_spatial_shape;

                    if (node.has_ng_input("OutSize"))
                    {
                        attrs.shape_calculation_mode = ShapeCalcMode::sizes;
                        auto hw_shape = node.get_ng_input("OutSize");
                        const auto shape_of_x = std::make_shared<opset6::ShapeOf>(x);
                        auto shape_begin = opset6::Constant::create(element::i64, {1}, {0});
                        auto shape_end = opset6::Constant::create(element::i64, Shape{1}, {2});
                        auto nc_node =
                            std::make_shared<opset6::StridedSlice>(shape_of_x,
                                                                   shape_begin,
                                                                   shape_end,
                                                                   std::vector<int64_t>{0},
                                                                   std::vector<int64_t>{0});
                        target_spatial_shape = std::make_shared<opset6::Concat>(
                            OutputVector{nc_node,
                                         std::make_shared<opset6::Convert>(hw_shape, element::i64)},
                            0);
                        scales = calculate_scales_based_on_sizes(x, target_spatial_shape);
                    }
                    else if (out_w <= 0 || out_h <= 0)
                    {
                        attrs.shape_calculation_mode = ShapeCalcMode::scales;
                        target_spatial_shape =
                            calculate_output_shape_based_on_scales(x, scale, scales);
                    }
                    else
                    {
                        attrs.shape_calculation_mode = ShapeCalcMode::sizes;
                        target_spatial_shape = extract_out_sizes(x, {out_h, out_w});
                        scales = calculate_scales_based_on_sizes(x, target_spatial_shape);
                    }

                    bool align_corners = node.get_attribute<bool>("align_corners");
                    int32_t align_mode = node.get_attribute<int32_t>("align_mode");

                    if (mode == InterpolateMode::nearest)
                    {
                        attrs.coordinate_transformation_mode = CoordinateTransformMode::asymmetric;
                    }
                    else if (!align_corners && align_mode == 1)
                    {
                        attrs.coordinate_transformation_mode = CoordinateTransformMode::asymmetric;
                    }
                    else if (!align_corners && align_mode == 0)
                    {
                        attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
                    }
                    else if (align_corners)
                    {
                        attrs.coordinate_transformation_mode =
                            CoordinateTransformMode::align_corners;
                    }

                    attrs.nearest_mode = Nearest_mode::simple;
                    attrs.antialias = false;
                    attrs.pads_begin = {0, 0, 0, 0};
                    attrs.pads_end = {0, 0, 0, 0};

                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Interpolate>(
                            x, target_spatial_shape, scales, attrs)},
                        {"Out"});
                }

                NamedOutputs bilinear_interp_v2(const NodeContext& node)
                {
                    auto mode = ngraph::opset6::Interpolate::InterpolateMode::linear_onnx;
                    return interpolate(node, mode);
                }

                NamedOutputs nearest_interp_v2(const NodeContext& node)
                {
                    auto mode = ngraph::opset6::Interpolate::InterpolateMode::nearest;
                    return interpolate(node, mode);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph