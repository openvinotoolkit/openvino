// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/extractimagepatches.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {

// Non-overlapping patches (a ViT patch embedding) are a pure reshape: crop to whole patches like
// ExtractImagePatches VALID, split H and W into (patches, patch), and move each patch's (C, KH, KW)
// to the last axis. ExtractImagePatches has no dynamic-shape GPU implementation.
ov::Output<Node> non_overlapping_patches(const ov::Output<Node>& image, size_t IC, size_t KH, size_t KW) {
    using ov::op::v0::Constant;
    const auto shape = std::make_shared<ov::op::v3::ShapeOf>(image, ov::element::i64);
    const auto kernel = Constant::create(ov::element::i64,
                                         {2},
                                         std::vector<int64_t>{static_cast<int64_t>(KH), static_cast<int64_t>(KW)});
    const auto patches = std::make_shared<ov::op::v1::Divide>(gather_dims(shape, {2, 3}), kernel);
    const auto cropped = std::make_shared<ov::op::v8::Slice>(image,
                                                             Constant::create(ov::element::i64, {2}, {0, 0}),
                                                             std::make_shared<ov::op::v1::Multiply>(patches, kernel),
                                                             Constant::create(ov::element::i64, {2}, {1, 1}),
                                                             Constant::create(ov::element::i64, {2}, {2, 3}));
    const auto split_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather_dims(shape, {0, 1}),
                         gather_dims(patches, {0}),
                         Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(KH)}),
                         gather_dims(patches, {1}),
                         Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(KW)})},
        0);
    const auto split = std::make_shared<ov::op::v1::Reshape>(cropped, split_shape, false);
    const auto grouped =
        std::make_shared<ov::op::v1::Transpose>(split, Constant::create(ov::element::i64, {6}, {0, 2, 4, 1, 3, 5}));
    return std::make_shared<ov::op::v1::Reshape>(
        grouped,
        Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, static_cast<int64_t>(IC * KH * KW)}),
        true);
}

}  // namespace

// GGML_OP_IM2COL: unfold a 1D/2D convolution input into column patches (conv / vision models).
// The decoder exposes the conv params (strides/pads/dilations + is_2D) as a typed int vector.
OutputVector translate_im2col(const NodeContext& context) {
    // Builder graphs pass the kernel shape as an attribute instead of a placeholder kernel tensor.
    num_inputs_check(context, 1, 2);
    const bool kernel_input = context.get_input_size() == 2;

    auto params = context.get_attribute<std::vector<int32_t>>("im2col_params");
    FRONT_END_OP_CONVERSION_CHECK(params.size() >= 7, "IM2COL requires 7 params");
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    bool is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_input(kernel_input ? 1 : 0);
    const ov::Shape kernel_shape =
        kernel_input ? context.get_input(0).get_shape() : context.get_attribute<ov::Shape>("kernel_shape");

    const size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    const size_t KH = is_2D ? kernel_shape[2] : 1;
    const size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    if (!is_2D) {
        const auto image_shape = std::make_shared<ov::op::v3::ShapeOf>(image, ov::element::i64);
        auto image_reshape_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{
                gather_dims(image_shape, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{static_cast<int64_t>(IC), 1}),
                gather_dims(image_shape, {3})},
            0);
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, false);
    }

    // Older cgraph decoders expose ggml_im2col's dst_type as output_type.
    const auto output_type = context.get_attribute<ov::element::Type>(
        "dst_type",
        context.get_attribute<ov::element::Type>("output_type", ov::element::dynamic));
    FRONT_END_OP_CONVERSION_CHECK(output_type.is_static(), "IM2COL requires 'dst_type'");

    if (is_2D && stride_h == static_cast<int32_t>(KH) && stride_w == static_cast<int32_t>(KW) && pad_h == 0 &&
        pad_w == 0 && dil_h == 1 && dil_w == 1) {
        res = non_overlapping_patches(image, IC, KH, KW);
    } else {
        const ov::Shape patch_sizes = {KH, KW};
        const ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
        const ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

        auto pads_begin =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
        auto pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_begin, ov::op::PadMode::CONSTANT);
        auto patches =
            std::make_shared<ov::op::v3::ExtractImagePatches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

        auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
        auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

        auto reshape1_shape = ov::op::v0::Constant::create(
            ov::element::i64,
            {5},
            std::vector<int64_t>{0, 0, 0, static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
        auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, true);

        auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

        auto r2_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                     {4},
                                                     std::vector<int64_t>{0, 0, 0, static_cast<int64_t>(IC * KH * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, true);

        if (!is_2D) {
            auto final_reshape_shape = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                 get_dimensions(t1, {0, 2}),
                                 ov::op::v0::Constant::create(ov::element::i64,
                                                              {1},
                                                              std::vector<int64_t>{static_cast<int64_t>(IC * KW)})},
                0);
            res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, false);
        }
    }

    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

// GGML_OP_IM2COL_3D: unfold a 3D convolution input into column patches. "im2col_params" is ggml's
// {s0, s1, s2, p0, p1, p2, d0, d1, d2, IC} (0 = width, 2 = depth); the kernel operand supplies KD/KH/KW.
OutputVector translate_im2col_3d(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto params = context.get_attribute<std::vector<int32_t>>("im2col_params");
    FRONT_END_OP_CONVERSION_CHECK(params.size() >= 10, "IM2COL_3D requires 10 params");
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t s2 = params[2];
    int32_t p0 = params[3];
    int32_t p1 = params[4];
    int32_t p2 = params[5];
    int32_t d0 = params[6];
    int32_t d1 = params[7];
    int32_t d2 = params[8];
    int32_t IC = params[9];
    const auto output_type = context.get_attribute<ov::element::Type>("dst_type", ov::element::f32);

    ov::Output<Node> image = context.get_input(1);
    const ov::Shape kernel_shape = context.get_input(0).get_shape();
    const ov::Shape image_shape = image.get_shape();

    const size_t KD = kernel_shape[1];
    const size_t KH = kernel_shape[2];
    const size_t KW = kernel_shape[3];

    const size_t N = image_shape[0] / static_cast<size_t>(IC);
    const size_t ID = image_shape[1];
    const size_t IH = image_shape[2];
    const size_t IW = image_shape[3];

    const size_t OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    const size_t OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    const size_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    if (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    const size_t IH_pad = IH + 2 * p1;
    const size_t IW_pad = IW + 2 * p0;

    auto image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    auto image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, false);

    auto pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pads_end = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_end, ov::op::PadMode::CONSTANT);

    const ov::Shape patch_sizes = {KH, KW};
    const ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    const ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::OutputVector kd_slices;
    kd_slices.reserve(KD);

    auto perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    auto reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    auto r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    auto r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    auto step_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    auto axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    for (size_t ikd = 0; ikd < KD; ++ikd) {
        auto start_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        auto stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        auto depth_slice = std::make_shared<ov::op::v8::Slice>(pad_3d, start_c, stop_c, step_c, axes_c);
        auto depth_slice_trans = std::make_shared<ov::op::v1::Transpose>(depth_slice, perm_nod);
        auto depth_slice_4d = std::make_shared<ov::op::v1::Reshape>(depth_slice_trans, reshape_4d_shape, false);

        auto patches = std::make_shared<ov::op::v3::ExtractImagePatches>(
            depth_slice_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, false);
        auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        auto r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, false);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    if (KD == 1) {
        res = kd_slices[0];
    } else {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    auto final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, false);

    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
