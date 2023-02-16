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
OutputVector base_translate_upsample2d(const NodeContext& context, v4::Interpolate::InterpolateMode interpolate_mode) {
    num_inputs_check(context, 3, 4);
    auto data = context.get_input(0);
    std::vector<size_t> pad{0};
    auto size_mode = v4::Interpolate::ShapeCalcMode::SIZES;
    bool align_corners = false;
    int scale_id = 2;
    if (interpolate_mode != v4::Interpolate::InterpolateMode::NEAREST) {
        scale_id = 3;
        if (!context.input_is_none(2)) {
            align_corners = context.const_input<bool>(2);
        }
    }
    auto target_axes = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int>({2, 3}));
    auto scales =
        context.mark_node(std::make_shared<v0::Constant>(element::f32, Shape{2}, std::vector<double>({1, 1})));
    auto output_sizes =
        context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int>({1, 1})));
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
        if (align_corners) {
            attrs.coordinate_transformation_mode = v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        }
    }
    return {context.mark_node(std::make_shared<v4::Interpolate>(data, output_sizes, scales, target_axes, attrs))};
};
}  // namespace

OutputVector translate_upsample_bilinear2d(NodeContext& context) {
    return base_translate_upsample2d(context, v4::Interpolate::InterpolateMode::LINEAR_ONNX);
};

OutputVector translate_upsample_nearest2d(NodeContext& context) {
    return base_translate_upsample2d(context, v4::Interpolate::InterpolateMode::NEAREST);
};

OutputVector translate_upsample_bicubic2d(NodeContext& context) {
    return base_translate_upsample2d(context, v4::Interpolate::InterpolateMode::CUBIC);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov