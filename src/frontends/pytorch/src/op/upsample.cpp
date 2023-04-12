// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector base_translate_upsample(const NodeContext& context,
                                     v4::Interpolate::InterpolateMode interpolate_mode,
                                     size_t dims) {
    num_inputs_check(context, 1, 4);
    auto data = context.get_input(0);
    std::vector<size_t> pad(dims, 0);
    auto size_mode = v4::Interpolate::ShapeCalcMode::SIZES;
    bool align_corners = false;
    int scale_id = 2;
    if (interpolate_mode != v4::Interpolate::InterpolateMode::NEAREST) {
        scale_id = 3;
        if (!context.input_is_none(2)) {
            align_corners = context.const_input<bool>(2);
        }
    }
    std::vector<int> spatial_axes;
    if (dims == 1) {
        spatial_axes = {2};
    } else if (dims == 2) {
        spatial_axes = {2, 3};
    } else if (dims == 3) {
        spatial_axes = {2, 3, 4};
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported number of dimensions in upsample");
    }
    auto target_axes = std::make_shared<v0::Constant>(element::i32, Shape{spatial_axes.size()}, spatial_axes);
    auto scales =
        context.mark_node(std::make_shared<v0::Constant>(element::f32, Shape{dims}, std::vector<double>(dims, 1)));
    auto output_sizes =
        context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{dims}, std::vector<int>(dims, 1)));
    if (context.input_is_none(1)) {
        FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(scale_id), "Scale or Output size should be provided");
        auto spatial_scales = context.get_input(scale_id);

        size_mode = v4::Interpolate::ShapeCalcMode::SCALES;
        scales = context.mark_node(std::make_shared<v1::Multiply>(spatial_scales, scales));
    } else {
        auto out_sizes = context.get_input(1);
        output_sizes = context.mark_node(std::make_shared<v1::Multiply>(out_sizes, output_sizes));
    }
    auto attrs = v4::Interpolate::InterpolateAttrs(interpolate_mode, size_mode, pad, pad);
    attrs.coordinate_transformation_mode = v4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = v4::Interpolate::NearestMode::FLOOR;
    if (attrs.mode != v4::Interpolate::InterpolateMode::NEAREST) {
        attrs.coordinate_transformation_mode = v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        if (align_corners) {
            attrs.coordinate_transformation_mode = v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        }
    }
    return {context.mark_node(std::make_shared<v4::Interpolate>(data, output_sizes, scales, target_axes, attrs))};
};
}  // namespace

OutputVector translate_upsample_linear1d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::LINEAR_ONNX, 1);
};

OutputVector translate_upsample_bilinear2d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::LINEAR_ONNX, 2);
};

OutputVector translate_upsample_trilinear3d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::LINEAR_ONNX, 3);
};

OutputVector translate_upsample_nearest1d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::NEAREST, 1);
};

OutputVector translate_upsample_nearest2d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::NEAREST, 2);
};

OutputVector translate_upsample_nearest3d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::NEAREST, 3);
};

// bicubic is only supported for 2d in pytorch
OutputVector translate_upsample_bicubic2d(const NodeContext& context) {
    return base_translate_upsample(context, v4::Interpolate::InterpolateMode::CUBIC, 2);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov