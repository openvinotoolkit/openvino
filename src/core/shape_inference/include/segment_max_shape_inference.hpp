// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <optional>

#include "openvino/op/segment_max.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SegmentMax* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 || input_shapes.size() == 3);

    // validate shape of data input
    const auto& data_shape = input_shapes[0];
    const auto is_data_shape_rank_static = data_shape.rank().is_static();
    if (is_data_shape_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, data_shape.size() > 0, "The data input cannot be a scalar.");
    }

    // validate segment_ids input
    const auto& segment_ids_shape = input_shapes[1];
    const auto is_segment_ids_rank_static = segment_ids_shape.rank().is_static();
    if (is_segment_ids_rank_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               segment_ids_shape.size() == 1,
                               "segment_ids must be a 1D input. Got: ",
                               segment_ids_shape);
        if (is_data_shape_rank_static) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   data_shape[0].compatible(segment_ids_shape[0]),
                                   "The number of elements in segment_ids must match the first dimension of data.");
        }
    }
    const auto segment_ids = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor);
    if (segment_ids) {
        NODE_VALIDATION_CHECK(op,
                              std::is_sorted(segment_ids->begin(), segment_ids->end()),
                              "segment_ids must be sorted.");
    }

    // validate num_segments input
    const auto num_segments_available = op->inputs().size() == 3;
    std::optional<TRShape> num_segments;
    if (num_segments_available) {
        num_segments = get_input_const_data_as_shape<TRShape>(op, 2, tensor_accessor);
    }

    if (num_segments_available) {
        const auto& num_segments_shape = input_shapes[2];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               num_segments_shape.rank().compatible(0),
                               "num_segments must be a scalar input. Got: ",
                               num_segments_shape);
    }

    if (!is_data_shape_rank_static) {
        return {PartialShape::dynamic()};
    }
    using TDim = typename TShape::value_type;
    auto output_shapes = std::vector<TRShape>{data_shape};
    auto& output_shape = output_shapes[0];
    if (num_segments) {
        output_shape[0] = TDim((*num_segments)[0]);
    } else if (segment_ids && !num_segments_available) {
        output_shape[0] = TDim(segment_ids->back() + 1);
    } else {
        // if num_segments input is provided but the value is unknown, the first dimension should be dynamic
        output_shape[0] = Dimension::dynamic();
    }
    return output_shapes;
}
}  // namespace v16
}  // namespace op
}  // namespace ov
