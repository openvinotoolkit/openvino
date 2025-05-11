// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compare.hpp"
#include "dimension_util.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const ReorgYolo* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    using namespace ov::util;

    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();
    const auto stride = op->get_strides().front();

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    if (input_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shape.size() == 4, "[N, C, H, W] input shape is required.");

        const auto stride_sq = typename T::value_type(stride * stride);
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            input_shape[1].is_dynamic() || cmp::ge(input_shape[1].get_length(), stride_sq.get_length()),
            "For [N, C, H, W] input shape, C >= (stride*stride) is required.");

        output_shape.reserve(input_shape.size());
        auto out_it = std::copy(input_shape.begin(), input_shape.begin() + 2, std::back_inserter(output_shape));
        for (size_t i = 2; i < input_shape.size(); ++i, ++out_it) {
            auto d = input_shape[i] / stride;
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   !dim::is_empty(d),
                                   "For [N, C, H, W] input shape, H and W should be divisible by stride.");
            out_it = std::move(d);
        }
        output_shape[1] *= stride_sq;
    } else {
        output_shape = PartialShape::dynamic(input_rank);
    }
    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
