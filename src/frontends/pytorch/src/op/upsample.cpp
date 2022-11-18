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
    std::vector<size_t> axes{2, 3};
    std::vector<double> default_scales{1, 1};
    std::vector<int64_t> default_sizes{1, 1};
    bool align_corners = false;
    int scale_id = 1;
    if (interpolate_mode == opset8::Interpolate::InterpolateMode::LINEAR_ONNX) {
        scale_id = 2;
        if (context.input_is_none(1)) {
            align_corners = context.const_input<bool>(1);
        }
    }
    auto target_axes = context.mark_node(opset8::Constant::create(element::i64, Shape(axes.size()), axes));

    auto size_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    auto scales = context.mark_node(opset8::Constant::create(element::f32, Shape({2}), default_scales));
    auto output_sizes = context.mark_node(opset8::Constant::create(element::i64, Shape({2}), default_sizes));

    if (context.input_is_none(1)) {
        FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(2), "Scale or Output size should be provided");
        auto spatial_scales = context.const_input<std::vector<double>>(2);
        if (spatial_scales.size() == 1) {
            spatial_scales.push_back(spatial_scales[0]);
        }
        size_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
        scales =
            context.mark_node(opset8::Constant::create(element::f32, Shape{spatial_scales.size()}, spatial_scales));
    } else {
        auto out_sizes = context.const_input<std::vector<int64_t>>(1);
        if (out_sizes.size() == 1) {
            out_sizes.push_back(out_sizes[0]);
        }
        output_sizes = context.mark_node(opset8::Constant::create(element::i64, Shape({2}), out_sizes));
    }
    auto attrs = opset8::Interpolate::InterpolateAttrs(interpolate_mode, size_mode, pad, pad);

    if (attrs.mode == opset8::Interpolate::InterpolateMode::LINEAR_ONNX) {
        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        if (align_corners) {
            attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        }
    }
    return {context.mark_node(std::make_shared<opset8::Interpolate>(data, output_sizes, scales, target_axes, attrs))};
};

OutputVector translate_upsample2d_bilinear(NodeContext& context) {
    return translate_upsample2d(context, opset8::Interpolate::InterpolateMode::LINEAR_ONNX);
};

OutputVector translate_upsample2d_nearest(NodeContext& context) {
    return translate_upsample2d(context, opset8::Interpolate::InterpolateMode::NEAREST);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov