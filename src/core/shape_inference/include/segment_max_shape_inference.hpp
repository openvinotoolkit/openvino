// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

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
    const auto& data_shape = input_shapes[0];
    const auto& segment_ids_shape = input_shapes[1];
    const auto& num_segments_shape = input_shapes.size() == 3 ? input_shapes[2] : TShape{};

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           segment_ids_shape.rank().compatible(1),
                           "segment_ids must be a 1D input. Got: ",
                           segment_ids_shape);

    const auto num_segments_available = op->inputs().size() == 3;
    if (num_segments_available) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               num_segments_shape.rank().compatible(0),
                               "num_segments must be a scalar input. Got: ",
                               num_segments_shape);
    }

    if (data_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, data_shape.size() > 0, "The data input cannot be a scalar.");
        auto output_shapes = std::vector<TRShape>{data_shape};
        auto& output_shape = output_shapes[0];
        const auto num_segments = num_segments_available
                                      ? ov::op::get_input_const_data_as<TRShape, int64_t>(op, 2, tensor_accessor)
                                      : ov::optional<std::vector<int64_t>>{};

        // Try to use segment_ids to infer the first dimension
        if (segment_ids_shape.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   data_shape[0].compatible(segment_ids_shape[0]),
                                   "The number of elements in segment_ids must match the first dimension of data.");
            if (const auto segment_ids = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor)) {
                NODE_VALIDATION_CHECK(op,
                                      std::is_sorted(segment_ids->begin(), segment_ids->end()),
                                      "segment_ids must be sorted.");
                auto max_segment_id = *std::max_element(segment_ids->begin(), segment_ids->end());
                if (num_segments) {
                    NODE_VALIDATION_CHECK(
                        op,
                        static_cast<typename TShape::value_type>(max_segment_id + 1) == (*num_segments)[0],
                        "num_segments value (",
                        (*num_segments)[0],
                        ") is inconsistent with number of segments given in segment_ids (",
                        max_segment_id + 1,
                        ").");
                }
                output_shape[0] = max_segment_id + 1;
                return output_shapes;
            }
        }

        // Use num_segments to infer the first dimension in case segment_ids unavailable
        if (num_segments) {
            output_shape[0] = (*num_segments)[0];
            return output_shapes;
        }

        output_shape[0] = Dimension::dynamic();
        return output_shapes;
    } else {
        return {PartialShape::dynamic()};
    }
}
}  // namespace v16
}  // namespace op
}  // namespace ov
