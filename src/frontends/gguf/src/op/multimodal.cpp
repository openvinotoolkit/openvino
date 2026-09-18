// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {
std::shared_ptr<ov::op::v0::Constant> i64_const(std::vector<int64_t> v) {
    return ov::op::v0::Constant::create(ov::element::i64, {v.size()}, v);
}
}  // namespace

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
    const std::vector<int> axes{2, 3};
    auto sizes = context.get_attribute<bool>("resize_like", false)
                     ? get_dimensions(context.get_input(1), axes)->output(0)
                     : context.get_input(1);
    auto result = std::make_shared<Interpolate>(
        context.get_input(0),
        sizes,
        ov::op::v0::Constant::create(ov::element::f32, {axes.size()}, std::vector<float>(axes.size(), 1.f)),
        ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes),
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

OutputVector translate_win_part(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    using namespace ov::op;
    auto x = context.get_input(0);
    const auto window = context.get_attribute<int64_t>("window");
    FRONT_END_OP_CONVERSION_CHECK(window > 0 && x.get_partial_shape().rank() == 4, "Invalid window partition");
    const auto& c = i64_const;
    auto spatial = get_dimensions(x, {1, 2});
    auto padding = std::make_shared<v1::FloorMod>(std::make_shared<v1::Subtract>(c({window, window}), spatial),
                                                  c({window, window}));
    auto padded = std::make_shared<v1::Pad>(x,
                                            c({0, 0, 0, 0}),
                                            std::make_shared<v0::Concat>(OutputVector{c({0}), padding, c({0})}, 0),
                                            PadMode::CONSTANT);
    auto size = std::make_shared<v1::Divide>(get_dimensions(padded, {1, 2}), c({window, window}), true);
    auto split = std::make_shared<v8::Gather>(size, c({0}), c({0}));
    auto split_w = std::make_shared<v8::Gather>(size, c({1}), c({0}));
    auto pattern = std::make_shared<v0::Concat>(
        OutputVector{get_dimensions(x, {0}), split, c({window}), split_w, c({window}), get_dimensions(x, {3})},
        0);
    auto reshaped = std::make_shared<v1::Reshape>(padded, pattern, false);
    auto ordered = std::make_shared<v1::Transpose>(reshaped, c({0, 1, 3, 2, 4, 5}));
    auto output = std::make_shared<v1::Reshape>(
        ordered,
        std::make_shared<v0::Concat>(OutputVector{c({-1, window, window}), get_dimensions(x, {3})}, 0),
        false);
    return rename_outputs_with_suffix({output}, context.get_name());
}

OutputVector translate_win_unpart(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    using namespace ov::op;
    auto x = context.get_input(0), reference = context.get_input(1);
    const auto window = context.get_attribute<int64_t>("window");
    FRONT_END_OP_CONVERSION_CHECK(window > 0, "Invalid window unpartition");
    const auto& c = i64_const;
    auto spatial = get_dimensions(reference, {1, 2});
    auto blocks = std::make_shared<v1::Divide>(std::make_shared<v1::Add>(spatial, c({window - 1, window - 1})),
                                               c({window, window}),
                                               true);
    auto h = std::make_shared<v8::Gather>(blocks, c({0}), c({0}));
    auto w = std::make_shared<v8::Gather>(blocks, c({1}), c({0}));
    auto pattern = std::make_shared<v0::Concat>(
        OutputVector{get_dimensions(reference, {0}), h, w, c({window, window}), get_dimensions(x, {3})},
        0);
    auto ordered =
        std::make_shared<v1::Transpose>(std::make_shared<v1::Reshape>(x, pattern, false), c({0, 1, 3, 2, 4, 5}));
    auto padded_shape =
        std::make_shared<v0::Concat>(OutputVector{get_dimensions(reference, {0}),
                                                  std::make_shared<v1::Multiply>(blocks, c({window, window})),
                                                  get_dimensions(x, {3})},
                                     0);
    auto padded = std::make_shared<v1::Reshape>(ordered, padded_shape, false);
    return rename_outputs_with_suffix({std::make_shared<v8::Slice>(padded, c({0, 0}), spatial, c({1, 1}), c({1, 2}))},
                                      context.get_name());
}

OutputVector translate_get_rel_pos(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    if (context.get_input_size() == 1) {
        const auto side = context.get_attribute<int64_t>("q_size");
        FRONT_END_OP_CONVERSION_CHECK(side > 0, "Invalid relative position grid");
        std::vector<int32_t> indices(size_t(side * side));
        for (int64_t q = 0; q < side; ++q)
            for (int64_t k = 0; k < side; ++k)
                indices[size_t(q * side + k)] = int32_t(q - k + side - 1);
        auto ids = ov::op::v0::Constant::create(ov::element::i32, {1, size_t(side), size_t(side)}, indices);
        auto table = std::make_shared<ov::op::v1::Reshape>(
            context.get_input(0),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{2 * side - 1, -1}),
            false);
        auto output =
            std::make_shared<ov::op::v8::Gather>(table, ids, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        return rename_outputs_with_suffix({std::make_shared<ov::op::v0::Convert>(output, ov::element::f16)},
                                          context.get_name());
    }
    FRONT_END_OP_CONVERSION_CHECK(context.get_op_case() == 1, "Indexed relative positions require case 1");
    using namespace ov::op;
    auto table = context.get_input(0), indices = context.get_input(1);
    const auto& c = i64_const;
    // SAM decomposed relative positions: resize the distance table before gathering.
    auto length =
        std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(get_dimensions(indices, {2}), c({2})), c({1}));
    auto data = std::make_shared<v1::Reshape>(
        std::make_shared<v1::Transpose>(table, c({1, 0})),
        std::make_shared<v0::Concat>(OutputVector{c({1}), get_dimensions(table, {1}), c({1, -1})}, 0),
        false);
    v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = v4::Interpolate::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = v4::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    auto resized = std::make_shared<v4::Interpolate>(data,
                                                     length,
                                                     v0::Constant::create(ov::element::f32, {1}, {1}),
                                                     c({3}),
                                                     attrs);
    auto rows = std::make_shared<v1::Reshape>(
        std::make_shared<v1::Transpose>(resized, c({0, 2, 3, 1})),
        std::make_shared<v0::Concat>(OutputVector{c({-1}), get_dimensions(table, {1})}, 0),
        false);
    auto ids = std::make_shared<v0::Squeeze>(indices, c({0}));
    return rename_outputs_with_suffix({std::make_shared<v8::Gather>(rows, ids, c({0}))}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
