// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/util/convert_color_to_nv12_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ConvertColorToNV12Base* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1, "RGBtoNV12/BGRtoNV12 shall have exactly one input");
    using namespace ov::util;

    const auto& shape_rgb = input_shapes[0];
    const auto rank = shape_rgb.rank();

    NODE_SHAPE_INFER_CHECK(op, input_shapes, rank.compatible(4), "RGB/BGR input shall have 4 dimensions (N, H, W, C)");

    if (rank.is_dynamic()) {
        if (op->is_single_plane()) {
            auto out_shape = TRShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 1};
            return {out_shape};
        } else {
            auto y_shape = TRShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 1};
            auto uv_shape = TRShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), 2};
            return {y_shape, uv_shape};
        }
    }

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           shape_rgb[3].compatible(3),
                           "RGB/BGR input number of channels should be equal to 3");

    NODE_SHAPE_INFER_CHECK(op, input_shapes, dim::is_divisible(shape_rgb[1], 2), "Image height must be even");
    NODE_SHAPE_INFER_CHECK(op, input_shapes, dim::is_divisible(shape_rgb[2], 2), "Image width must be even");

    if (op->is_single_plane()) {
        auto out_shape = TRShape{shape_rgb[0], shape_rgb[1], shape_rgb[2], 1};
        out_shape[1] *= 3;
        out_shape[1] /= 2;

        NODE_SHAPE_INFER_CHECK(op, input_shapes, !dim::is_empty(out_shape[1]), "Image height computation failed");

        return {out_shape};
    } else {
        auto y_shape = TRShape{shape_rgb[0], shape_rgb[1], shape_rgb[2], 1};
        auto uv_shape = TRShape{shape_rgb[0], shape_rgb[1], shape_rgb[2], 2};

        uv_shape[1] /= 2;
        uv_shape[2] /= 2;

        return {y_shape, uv_shape};
    }
}
}  // namespace op
}  // namespace ov
