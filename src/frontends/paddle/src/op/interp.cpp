// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset4.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace default_opset;

static std::shared_ptr<ov::Node> calculate_output_shape_based_on_scales(const Output<ov::Node>& data,
                                                                        const std::vector<float>& scale,
                                                                        Output<ov::Node>& scales,
                                                                        const int space_dim) {
    const size_t scale_size = static_cast<size_t>(space_dim + 2);
    FRONT_END_GENERAL_CHECK(scale.size() > 0 && scale.size() <= scale_size);

    std::vector<float> full_scales(scale_size, 1.0f);
    std::memcpy(&full_scales[scale_size - scale.size()], &scale[0], scale.size() * sizeof(float));
    scales = Constant::create<float>(element::f32, {scale_size}, full_scales);

    const auto shape_of_data = std::make_shared<Convert>(std::make_shared<ShapeOf>(data), scales.get_element_type());
    const auto multiply = std::make_shared<Multiply>(shape_of_data, scales);
    const auto output_shape = std::make_shared<Convert>(multiply, ov::element::i64);

    return output_shape;
}

static std::shared_ptr<ov::Node> calculate_scales_based_on_sizes(const Output<ov::Node>& data,
                                                                 const Output<ov::Node>& sizes) {
    const float epsilon = 1.0e-5f;
    const auto shape_of_data = std::make_shared<Convert>(std::make_shared<ShapeOf>(data), ov::element::f32);
    const auto converted_sizes = std::make_shared<Convert>(sizes, ov::element::f32);
    const auto divide = std::make_shared<Divide>(converted_sizes, shape_of_data);
    const auto eps_node = std::make_shared<Constant>(ov::element::f32, Shape{}, epsilon);
    const auto scales = std::make_shared<Add>(divide, eps_node);

    return scales;
}

static std::shared_ptr<ov::Node> extract_out_sizes(const Output<ov::Node>& data,
                                                   const std::vector<int64_t>& out_sizes) {
    const auto shape_of_x = std::make_shared<ShapeOf>(data);
    const auto shape_begin = Constant::create(element::i64, {1}, {0});
    const int end_idx = static_cast<int>(out_sizes.size());
    const auto shape_end = Constant::create(element::i64, Shape{1}, {-end_idx});
    const auto nc_node = std::make_shared<StridedSlice>(shape_of_x,
                                                        shape_begin,
                                                        shape_end,
                                                        std::vector<int64_t>{0},
                                                        std::vector<int64_t>{0});
    const auto hw_node = Constant::create<int64_t>(element::i64, Shape{out_sizes.size()}, out_sizes);
    return std::make_shared<Concat>(OutputVector{nc_node, hw_node}, 0);
}

// TODO support different data_layout #55170

static NamedOutputs interpolate(const NodeContext& node,
                                const Interpolate::InterpolateMode& mode,
                                const int space_dim) {
    const auto x = node.get_input("X");
    using InterpolateMode = Interpolate::InterpolateMode;
    using CoordinateTransformMode = Interpolate::CoordinateTransformMode;
    using Nearest_mode = Interpolate::NearestMode;
    using InterpolateAttrs = Interpolate::InterpolateAttrs;
    using ShapeCalcMode = Interpolate::ShapeCalcMode;

    InterpolateAttrs attrs;

    attrs.mode = mode;

    auto out_w = node.get_attribute<int>("out_w");
    auto out_h = node.get_attribute<int>("out_h");
    auto out_d = node.get_attribute<int>("out_d");
    auto scale = node.get_attribute<std::vector<float>>("scale");
    Output<Node> scales;
    Output<Node> target_spatial_shape;
    bool out_flag = out_w <= 0;
    if (space_dim == 2) {
        out_flag |= out_h <= 0;
    } else if (space_dim == 3) {
        out_flag |= out_h <= 0 || out_d <= 0;
    }

    if (node.has_input("SizeTensor") || node.has_input("OutSize")) {
        attrs.shape_calculation_mode = ShapeCalcMode::SIZES;
        const auto hw_shapes =
            node.has_input("SizeTensor") ? node.get_ng_inputs("SizeTensor") : node.get_ng_inputs("OutSize");
        const auto shape_of_x = std::make_shared<ShapeOf>(x);
        const auto shape_begin = Constant::create(element::i64, {1}, {0});
        const auto shape_end = Constant::create(element::i64, Shape{1}, {-space_dim});
        const auto nc_node = std::make_shared<StridedSlice>(shape_of_x,
                                                            shape_begin,
                                                            shape_end,
                                                            std::vector<int64_t>{0},
                                                            std::vector<int64_t>{0});
        OutputVector shapes{nc_node};
        const auto const_n1 = Constant::create(ov::element::i64, Shape{1}, {-1});
        for (auto node : hw_shapes) {
            shapes.push_back(std::make_shared<Reshape>(std::make_shared<Convert>(node, element::i64), const_n1, true));
        }
        target_spatial_shape = std::make_shared<Concat>(shapes, 0);
        scales = calculate_scales_based_on_sizes(x, target_spatial_shape);
    } else if (out_flag) {
        attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
        target_spatial_shape = calculate_output_shape_based_on_scales(x, scale, scales, space_dim);
    } else {
        attrs.shape_calculation_mode = ShapeCalcMode::SIZES;
        std::vector<int64_t> sizes;
        if (space_dim == 1)
            sizes = {out_w};
        else if (space_dim == 2)
            sizes = {out_h, out_w};
        else
            sizes = {out_d, out_h, out_w};

        target_spatial_shape = extract_out_sizes(x, sizes);
        scales = calculate_scales_based_on_sizes(x, target_spatial_shape);
    }

    const bool align_corners = node.get_attribute<bool>("align_corners");
    const int32_t align_mode = node.get_attribute<int32_t>("align_mode");

    if (mode == InterpolateMode::NEAREST) {
        attrs.coordinate_transformation_mode = CoordinateTransformMode::ASYMMETRIC;
    } else if (mode == InterpolateMode::CUBIC) {
        if (!align_corners) {
            attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
        } else {
            attrs.coordinate_transformation_mode = CoordinateTransformMode::ALIGN_CORNERS;
        }
    } else {
        if (!align_corners && align_mode == 1) {
            attrs.coordinate_transformation_mode = CoordinateTransformMode::ASYMMETRIC;
        } else if (!align_corners && align_mode == 0) {
            attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
        } else if (align_corners) {
            attrs.coordinate_transformation_mode = CoordinateTransformMode::ALIGN_CORNERS;
        }
    }

    attrs.nearest_mode = Nearest_mode::SIMPLE;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    return node.default_single_output_mapping(
        {std::make_shared<ov::opset4::Interpolate>(x, target_spatial_shape, scales, attrs)},
        {"Out"});
}

NamedOutputs linear_interp_v2(const NodeContext& node) {
    const auto mode = Interpolate::InterpolateMode::LINEAR_ONNX;
    return interpolate(node, mode, 1);
}

NamedOutputs bilinear_interp_v2(const NodeContext& node) {
    const auto mode = Interpolate::InterpolateMode::LINEAR_ONNX;
    return interpolate(node, mode, 2);
}

NamedOutputs trilinear_interp_v2(const NodeContext& node) {
    const auto mode = Interpolate::InterpolateMode::LINEAR_ONNX;
    return interpolate(node, mode, 3);
}

NamedOutputs nearest_interp_v2(const NodeContext& node) {
    const auto mode = Interpolate::InterpolateMode::NEAREST;
    return interpolate(node, mode, 2);
}

NamedOutputs bicubic_interp_v2(const NodeContext& node) {
    const auto mode = Interpolate::InterpolateMode::CUBIC;
    return interpolate(node, mode, 2);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
