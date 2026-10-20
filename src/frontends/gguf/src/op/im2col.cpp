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

}  // namespace ov::frontend::gguf::op
