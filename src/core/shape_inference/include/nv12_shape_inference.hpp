// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/util/convert_color_nv12_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ConvertColorNV12Base* op, const std::vector<TShape>& input_shapes) {
    const auto has_single_plane = input_shapes.size() == 1;
    NODE_VALIDATION_CHECK(op, has_single_plane || input_shapes.size() == 2);
    using TDim = typename std::iterator_traits<typename TShape::iterator>::value_type;
    using namespace ov::util;

    const auto& shape_y = input_shapes[0];
    const auto rank_y = shape_y.rank();

    NODE_SHAPE_INFER_CHECK(op, input_shapes, rank_y.compatible(4), "Y(UV) input shall have 4 dimensions (N, H, W,C)");

    auto output_shapes = std::vector<TRShape>{shape_y};
    auto& out_shape = output_shapes.front();

    if (rank_y.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               shape_y[3].compatible(1),
                               "YUV input number of channels should be equal to 1");
    } else {
        out_shape.resize(4);
    }

    if (has_single_plane) {
        out_shape[1] *= 2;
        out_shape[1] /= 3;
        NODE_SHAPE_INFER_CHECK(op, input_shapes, !dim::is_empty(out_shape[1]), "Image height shall be divisible by 3");
    } else {
        TRShape shape_uv = input_shapes[1];
        if (shape_uv.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   (shape_uv.size() == 4) && shape_uv[3].compatible(2),
                                   "UV input number of channels should be equal to 2");
            std::for_each(shape_uv.begin() + 1, shape_uv.end() - 1, [](TDim& d) {
                d *= 2;
            });
        }

        out_shape[3] = 2;
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TRShape::merge_into(out_shape, shape_uv),
                               "Y shape is inconsistent with UV");
    }
    out_shape[3] = 3;

    NODE_SHAPE_INFER_CHECK(op, input_shapes, dim::is_divisible(out_shape[1], 2), "Image height must be even");
    NODE_SHAPE_INFER_CHECK(op, input_shapes, dim::is_divisible(out_shape[2], 2), "Image width must be even");

    return output_shapes;
}
}  // namespace op
}  // namespace ov
