// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/experimental_detectron_detection_output.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(const ExperimentalDetectronDetectionOutput* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 3);

    const auto& rois_shape = input_shapes[0];
    const auto& deltas_shape = input_shapes[1];
    const auto& scores_shape = input_shapes[2];
    const auto& im_info_shape = input_shapes[3];

    auto& output_box_shape = output_shapes[0];
    auto& output_det_shape = output_shapes[1];
    auto& output_score_shape = output_shapes[2];

    output_box_shape.resize(2);
    output_det_shape.resize(1);
    output_score_shape.resize(1);

    const auto rois_shape_rank_is_static = rois_shape.rank().is_static();

    if (rois_shape_rank_is_static) {
        NODE_VALIDATION_CHECK(op, rois_shape.size() == 2, "Input rois rank must be equal to 2.");

        NODE_VALIDATION_CHECK(op,
                              rois_shape[1].compatible(4),
                              "The last dimension of the 'input_rois' input must be compatible with 4. "
                              "Got: ",
                              rois_shape[1]);
    }
    const auto deltas_shape_rank_is_static = deltas_shape.rank().is_static();
    if (deltas_shape_rank_is_static) {
        NODE_VALIDATION_CHECK(op, deltas_shape.size() == 2, "Input deltas rank must be equal to 2.");

        NODE_VALIDATION_CHECK(op,
                              deltas_shape[1].compatible(op->m_attrs.num_classes * 4),
                              "The last dimension of the 'input_deltas' input be compatible with "
                              "the value of the attribute 'num_classes' * 4. Got: ",
                              deltas_shape[1]);
    }
    const auto scores_shape_is_static = scores_shape.rank().is_static();
    if (scores_shape_is_static) {
        NODE_VALIDATION_CHECK(op, scores_shape.size() == 2, "Input scores rank must be equal to 2.");

        NODE_VALIDATION_CHECK(op,
                              scores_shape[1].compatible(op->m_attrs.num_classes),
                              "The last dimension of the 'input_scores' input must be compatible with"
                              "the value of the attribute 'num_classes'. Got: ",
                              scores_shape[1]);
    }

    NODE_VALIDATION_CHECK(op, im_info_shape.rank().compatible(2), "Input image info rank must be compatible with 2.");

    if (rois_shape_rank_is_static && deltas_shape_rank_is_static && scores_shape_is_static) {
        const auto& num_batches_rois = rois_shape[0];
        const auto& num_batches_deltas = deltas_shape[0];
        const auto& num_batches_scores = scores_shape[0];
        auto merge_res = rois_shape[0];

        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merge_res, num_batches_rois, num_batches_deltas) &&
                                  DimType::merge(merge_res, merge_res, num_batches_scores),
                              "The first dimension of inputs 'input_rois', 'input_deltas', "
                              "'input_scores' must be the compatible. input_rois batch: ",
                              num_batches_rois,
                              "; input_deltas batch: ",
                              num_batches_deltas,
                              "; input_scores batch: ",
                              num_batches_scores);
    }

    const auto& rois_num = op->m_attrs.max_detections_per_image;

    output_box_shape[0] = rois_num;
    output_box_shape[1] = 4;
    output_det_shape[0] = rois_num;
    output_score_shape[0] = rois_num;
}

}  // namespace v6
}  // namespace op
}  // namespace ov
