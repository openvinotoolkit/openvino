// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_upscale(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const int flags = context.get_attribute<int>("interpolation_mode", 1);
    FRONT_END_OP_CONVERSION_CHECK((flags & ~0x303) == 0 && (flags & 0xff) <= 2, "Unsupported interpolation mode");
    using Interpolate = ov::op::v4::Interpolate;
    Interpolate::InterpolateAttrs attrs;
    attrs.mode = (flags & 0xff) == 0   ? Interpolate::InterpolateMode::NEAREST
                 : (flags & 0xff) == 1 ? Interpolate::InterpolateMode::LINEAR
                                       : Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = (flags & 0x100)       ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                                           : (flags & 0xff) == 0 ? Interpolate::CoordinateTransformMode::ASYMMETRIC
                                                                 : Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
    attrs.antialias = (flags & 0x200) != 0;
    attrs.pads_begin = attrs.pads_end = {0, 0, 0, 0};
    auto sizes = context.get_attribute<bool>("resize_like", false)
                     ? get_dimensions(context.get_input(1), {2, 3})->output(0)
                     : context.get_input(1);
    auto result = std::make_shared<Interpolate>(context.get_input(0),
                                                sizes,
                                                ov::op::v0::Constant::create(ov::element::f32, {2}, {1, 1}),
                                                ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3}),
                                                attrs);
    return rename_outputs_with_suffix({result}, context.get_name());
}

OutputVector translate_unary_gelu_erf(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    return rename_outputs_with_suffix(
        {std::make_shared<ov::op::v7::Gelu>(context.get_input(0), ov::op::GeluApproximationMode::ERF)},
        context.get_name());
}

OutputVector translate_pool_2d(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto p = context.get_attribute<std::vector<int64_t>>("pool_params");
    FRONT_END_OP_CONVERSION_CHECK(p.size() == 7, "POOL_2D requires mode,kx,ky,sx,sy,px,py");
    FRONT_END_OP_CONVERSION_CHECK(p[1] > 0 && p[2] > 0 && p[3] > 0 && p[4] > 0 && p[5] >= 0 && p[6] >= 0,
                                  "Invalid POOL_2D parameters");
    ov::Shape kernel{size_t(p[2]), size_t(p[1])}, pads{size_t(p[6]), size_t(p[5])};
    ov::Strides strides{size_t(p[4]), size_t(p[3])};
    if (p[0] == 1) {
        return rename_outputs_with_suffix({std::make_shared<ov::op::v1::AvgPool>(context.get_input(0),
                                                                                 strides,
                                                                                 pads,
                                                                                 pads,
                                                                                 kernel,
                                                                                 false,
                                                                                 ov::op::RoundingType::FLOOR)},
                                          context.get_name());
    }
    FRONT_END_OP_CONVERSION_CHECK(p[0] == 0, "Unknown POOL_2D mode");
    return rename_outputs_with_suffix(
        {std::make_shared<ov::op::v1::MaxPool>(context.get_input(0), strides, pads, pads, kernel)},
        context.get_name());
}

OutputVector translate_conv_2d(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto p = context.get_attribute<std::vector<int64_t>>("conv_params");
    FRONT_END_OP_CONVERSION_CHECK(p.size() == 6, "CONV_2D requires sx,sy,px,py,dx,dy");
    FRONT_END_OP_CONVERSION_CHECK(p[0] > 0 && p[1] > 0 && p[2] >= 0 && p[3] >= 0 && p[4] > 0 && p[5] > 0,
                                  "Invalid CONV_2D parameters");
    auto w = context.get_input(0), x = context.get_input(1);
    if (w.get_element_type() != ov::element::f32)
        w = std::make_shared<ov::op::v0::Convert>(w, ov::element::f32);
    if (x.get_element_type() != ov::element::f32)
        x = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
    return rename_outputs_with_suffix(
        {std::make_shared<ov::op::v1::Convolution>(x,
                                                   w,
                                                   ov::Strides{size_t(p[1]), size_t(p[0])},
                                                   ov::CoordinateDiff{p[3], p[2]},
                                                   ov::CoordinateDiff{p[3], p[2]},
                                                   ov::Strides{size_t(p[5]), size_t(p[4])})},
        context.get_name());
}

}  // namespace ov::frontend::gguf::op
