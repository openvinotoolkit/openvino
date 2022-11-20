// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_upsample2d(NodeContext& context, opset8::Interpolate::InterpolateMode interpolate_mode) {
    auto data = context.get_input(0);
    std::vector<size_t> pad{0};
    auto size_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    bool align_corners = false;
    int scale_id = 2;
    if (interpolate_mode == opset8::Interpolate::InterpolateMode::LINEAR_ONNX) {
        scale_id = 3;
        if (!context.input_is_none(2)) {
            align_corners = context.const_input<bool>(2);
        }
    }
    auto target_axes = std::make_shared<opset8::Constant>(element::i32, Shape{2}, std::vector<int>({2, 3}));
    auto scales = context.mark_node(std::make_shared<opset8::Constant>(element::f32, Shape{2}, std::vector<double>({1, 1})));
    auto output_sizes = std::make_shared<opset8::Constant>(element::i32, Shape{2}, std::vector<int>({1, 1}));
    if (context.input_is_none(1)) {
        FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(scale_id), "Scale or Output size should be provided");
        auto spatial_scales = context.const_input<std::vector<double>>(scale_id);
        if (spatial_scales.size() == 1) {
            spatial_scales.push_back(spatial_scales[0]);
        }
        size_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
        scales = std::make_shared<opset8::Constant>(element::f32, Shape{spatial_scales.size()}, spatial_scales);
    } else {
        auto out_sizes = context.const_input<std::vector<int64_t>>(1);
        if (out_sizes.size() == 1) {
            out_sizes.push_back(out_sizes[0]);
        }
        output_sizes = std::make_shared<opset8::Constant>(element::i64, Shape({2}), out_sizes);
    }
    auto attrs = opset8::Interpolate::InterpolateAttrs(interpolate_mode, size_mode, pad, pad);
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::FLOOR;
    if (attrs.mode == opset8::Interpolate::InterpolateMode::LINEAR_ONNX) {
        if (align_corners) {
            attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        }
    }
    return {context.mark_node(std::make_shared<opset8::Interpolate>(data, output_sizes, scales, target_axes, attrs))};
};

OutputVector translate_upsample_bilinear2d(NodeContext& context) {
    return translate_upsample2d(context, opset8::Interpolate::InterpolateMode::LINEAR_ONNX);
};

OutputVector translate_upsample_nearest2d(NodeContext& context) {
    return translate_upsample2d(context, opset8::Interpolate::InterpolateMode::NEAREST);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov