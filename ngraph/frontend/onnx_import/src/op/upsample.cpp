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
                    NGRAPH_CHECK((data.get_partial_shape().rank().is_static() &&
                                  data.get_partial_shape().rank().get_length() == 4),
                                 "Upsample v1 input tensor is required to be 4D.");
                    const auto rank = data.get_partial_shape().rank().get_length();

                    std::vector<float> scales(rank, 1.f);
                    scales[rank - 1] = width_scale;
                    scales[rank - 2] = height_scale;

                    const auto scales_const = default_opset::Constant::create(
                        ngraph::element::f32, Shape({scales.size()}), scales);

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales_const);
                    const auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);

                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales_const, get_attributes(mode))};
                }

            } // namespace set_1

            namespace set_7
            {
                OutputVector upsample(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);

                    const auto data_shape = data.get_partial_shape();

                    const auto scales = node.get_attribute_value<std::vector<float>>("scales");
                    const auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    check_mode_support(node, mode, version_7);

                    const auto scales_const = default_opset::Constant::create(
                        ngraph::element::f32, Shape({scales.size()}), scales);

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales_const);
                    const auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);

                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales_const, get_attributes(mode))};
                }

            } // namespace set_7

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
                    check_mode_support(node, mode, version_9);

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of Scales input is required to be static.");

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
                            data, output_shape_const, scales, get_attributes(mode))};
                    }

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    const auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);

                    return {std::make_shared<default_opset::Interpolate>(
                        data, output_shape, scales, get_attributes(mode))};
                }

            } // namespace set_9

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
