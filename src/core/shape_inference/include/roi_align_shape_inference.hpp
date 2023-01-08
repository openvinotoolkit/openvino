// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/roi_align.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {
template <class OpType, class ShapeType>
void infer_roi_align_shape(const OpType* op,
                           const std::vector<ShapeType>& input_shapes,
                           std::vector<ShapeType>& output_shapes) {
    using DimType = typename std::iterator_traits<typename ShapeType::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);

    const auto& input_ps = input_shapes[0];
    const auto& rois_ps = input_shapes[1];
    const auto& batch_indices_ps = input_shapes[2];

    const auto rois_ps_rank = rois_ps.rank();
    const auto input_ps_rank = input_ps.rank();
    const auto batch_indices_ps_rank = batch_indices_ps.rank();

    NODE_VALIDATION_CHECK(op, input_ps_rank.compatible(4), "Expected a 4D tensor for the input data. Got: ", input_ps);

    NODE_VALIDATION_CHECK(op, rois_ps_rank.compatible(2), "Expected a 2D tensor for the ROIs input. Got: ", rois_ps);

    NODE_VALIDATION_CHECK(op,
                          batch_indices_ps_rank.compatible(1),
                          "Expected a 1D tensor for the batch indices input. Got: ",
                          batch_indices_ps);

    if (rois_ps_rank.is_static()) {
        const auto& rois_second_dim = rois_ps[1];
        NODE_VALIDATION_CHECK(op,
                              rois_second_dim.compatible(4),
                              "The second dimension of ROIs input should contain box coordinates. ",
                              "op dimension is expected to be equal to 4. Got: ",
                              rois_second_dim);

        if (batch_indices_ps_rank.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  rois_ps[0].compatible(batch_indices_ps[0]),
                                  "The first dimension of ROIs input must be equal to the first dimension ",
                                  "of the batch indices input. Got: ",
                                  rois_ps[0],
                                  " and: ",
                                  batch_indices_ps[0]);
        }
    }

    auto& output_shape = output_shapes[0];
    output_shape.resize(4);
    output_shape[1] = input_ps_rank.is_static() ? input_ps[1] : -1;
    output_shape[2] = op->get_pooled_h();
    output_shape[3] = op->get_pooled_w();

    // if either of those 2 dimensions is static its value will be used
    // for the first dimension of the output shape - 'NUM_ROIS'
    if (rois_ps_rank.is_static() && batch_indices_ps_rank.is_static()) {
        OPENVINO_ASSERT(DimType::merge(output_shape[0], batch_indices_ps[0], rois_ps[0]));
    } else if (rois_ps_rank.is_static()) {
        output_shape[0] = rois_ps[0];
    } else if (batch_indices_ps_rank.is_static()) {
        output_shape[0] = batch_indices_ps[0];
    } else {
        output_shape[0] = Dimension::dynamic();
    }
}
template <class T>
void shape_infer(const ov::op::v3::ROIAlign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    ov::op::v3::infer_roi_align_shape(op, input_shapes, output_shapes);
}

}  // namespace v3
namespace v9 {

template <class T>
void shape_infer(const ov::op::v9::ROIAlign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);

    ov::op::v3::infer_roi_align_shape(op, input_shapes, output_shapes);
}

}  // namespace v9
}  // namespace op
}  // namespace ov
