// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "op/upsample.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                constexpr unsigned version_1{1};
                constexpr unsigned version_7{7};
                constexpr unsigned version_9{9};

                void check_mode_support(const onnx_import::Node& node,
                                        const std::string& mode,
                                        const unsigned op_version)
                {
                    const std::unordered_set<std::string> modes_v1 = {"nearest", "bilinear"};
                    const std::unordered_set<std::string> modes_v7 = {"nearest", "linear"};
                    const auto& supported_modes = op_version < version_7 ? modes_v1 : modes_v7;

                    if (std::find(supported_modes.cbegin(), supported_modes.cend(), mode) ==
                        supported_modes.cend())
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_modes)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        CHECK_VALID_NODE(node,
                                         false,
                                         mode,
                                         " - this type of interpolation mode is not supported."
                                         " Choose one of the following modes: ",
                                         supported_modes_str);
                    }
                }

                default_opset::Interpolate::InterpolateAttrs get_attributes(std::string mode)
                {
                    using InterpolateMode = default_opset::Interpolate::InterpolateMode;
                    using Transform_mode = default_opset::Interpolate::CoordinateTransformMode;
                    using ShapeCalcMode = default_opset::Interpolate::ShapeCalcMode;

                    const auto interpolate_mode =
                        (mode == "linear" || mode == "bilinear" ? InterpolateMode::linear_onnx
                                                                : InterpolateMode::nearest);

                    std::vector<size_t> pad{0};
                    auto attrs = default_opset::Interpolate::InterpolateAttrs(
                        interpolate_mode, ShapeCalcMode::scales, pad, pad);

                    if (attrs.mode == InterpolateMode::linear_onnx)
                        attrs.coordinate_transformation_mode = Transform_mode::asymmetric;

                    return attrs;
                }

                OutputVector create_upsample_subgraph(const Output<ngraph::Node>& data,
                                                      const Output<ngraph::Node>& scales,
                                                      const std::string& mode)
                {
                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    const auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);

                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales, get_attributes(mode))};
                }

            } // namespace

            namespace set_1
            {
                OutputVector upsample(const onnx_import::Node& node)
                {
                    const auto height_scale = node.get_attribute_value<float>("height_scale");
                    const auto width_scale = node.get_attribute_value<float>("width_scale");
                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    check_mode_support(node, mode, version_1);

                    const auto data = node.get_ng_inputs().at(0);

                    static const std::string expectation{"Input tensor is required to be 4D."};
                    const auto rank = data.get_partial_shape().rank();
                    CHECK_VALID_NODE(node, rank.is_static(), expectation);
                    const auto rank_size = rank.get_length();
                    CHECK_VALID_NODE(node, rank_size == 4, expectation);

                    std::vector<float> scales(rank_size, 1.f);
                    scales[rank_size - 1] = width_scale;
                    scales[rank_size - 2] = height_scale;

                    const auto scales_const = default_opset::Constant::create(
                        ngraph::element::f32, Shape({scales.size()}), scales);

                    return create_upsample_subgraph(data, scales_const, mode);
                }

            } // namespace set_1

            namespace set_7
            {
                OutputVector upsample(const onnx_import::Node& node)
                {
                    const auto scales = node.get_attribute_value<std::vector<float>>("scales");
                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    check_mode_support(node, mode, version_7);

                    const auto data = node.get_ng_inputs().at(0);

                    const auto rank = data.get_partial_shape().rank();
                    CHECK_VALID_NODE(node,
                                     rank.is_static() &&
                                         static_cast<int64_t>(scales.size()) == rank.get_length(),
                                     "Input tensor's rank is required to be the same as number of "
                                     "elements of 'scales' attribute.");

                    const auto scales_const = default_opset::Constant::create(
                        ngraph::element::f32, Shape({scales.size()}), scales);

                    return create_upsample_subgraph(data, scales_const, mode);
                }

            } // namespace set_7

            namespace set_9
            {
                OutputVector upsample(const onnx_import::Node& node)
                {
                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    check_mode_support(node, mode, version_9);

                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);
                    const auto& scales = inputs.at(1);

                    const auto& data_shape = data.get_partial_shape();
                    const auto& scales_shape = scales.get_partial_shape();
                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of Scales input is required to be static.");

                    return create_upsample_subgraph(data, scales, mode);
                }

            } // namespace set_9

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
