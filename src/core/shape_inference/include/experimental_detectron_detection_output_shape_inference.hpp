// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/experimental_detectron_detection_output.hpp"
#include "utils.hpp"
namespace ov {
namespace op {
namespace v6 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ExperimentalDetectronDetectionOutput* op,
                                 const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);
    using TDim = typename TShape::value_type;

    const auto& rois_shape = input_shapes[0];
    const auto& deltas_shape = input_shapes[1];
    const auto& scores_shape = input_shapes[2];
    const auto& im_info_shape = input_shapes[3];

    const auto rois_shape_rank_is_static = rois_shape.rank().is_static();

    if (rois_shape_rank_is_static) {
        NODE_VALIDATION_CHECK(op, rois_shape.size() == 2, "Input rois rank must be equal to 2.");

        NODE_VALIDATION_CHECK(op,
                              rois_shape[1].compatible(4),
                              "The last dimension of the 'input_rois' input must be compatible with 4. Got: ",
                              rois_shape[1]);
    }

    const auto& attrs = op->get_attrs();
    const auto deltas_shape_rank_is_static = deltas_shape.rank().is_static();
    if (deltas_shape_rank_is_static) {
        NODE_VALIDATION_CHECK(op, deltas_shape.size() == 2, "Input deltas rank must be equal to 2.");

        NODE_VALIDATION_CHECK(op,
                              deltas_shape[1].compatible(attrs.num_classes * 4),
                              "The last dimension of the 'input_deltas' input be compatible with "
                              "the value of the attribute 'num_classes' * 4. Got: ",
                              deltas_shape[1]);
    }
    const auto scores_shape_is_static = scores_shape.rank().is_static();
    if (scores_shape_is_static) {
        NODE_VALIDATION_CHECK(op, scores_shape.size() == 2, "Input scores rank must be equal to 2.");

        NODE_VALIDATION_CHECK(op,
                              scores_shape[1].compatible(attrs.num_classes),
                              "The last dimension of the 'input_scores' input must be compatible with"
                              "the value of the attribute 'num_classes'. Got: ",
                              scores_shape[1]);
    }

    NODE_VALIDATION_CHECK(op,
                          im_info_shape.compatible(TRShape{1, 3}),
                          "Input image info shape must be compatible with [1,3].");

    if (rois_shape_rank_is_static && deltas_shape_rank_is_static && scores_shape_is_static) {
        const auto& num_batches_rois = rois_shape[0];
        const auto& num_batches_deltas = deltas_shape[0];
        const auto& num_batches_scores = scores_shape[0];
        TDim merge_res;

        NODE_VALIDATION_CHECK(op,
                              TDim::merge(merge_res, num_batches_rois, num_batches_deltas) &&
                                  TDim::merge(merge_res, merge_res, num_batches_scores),
                              "The first dimension of inputs 'input_rois', 'input_deltas', "
                              "'input_scores' must be the compatible. input_rois batch: ",
                              num_batches_rois,
                              "; input_deltas batch: ",
                              num_batches_deltas,
                              "; input_scores batch: ",
                              num_batches_scores);
    }

    auto output_shapes = std::vector<TRShape>(3, TRShape{TDim(attrs.max_detections_per_image)});
    output_shapes[0].push_back(4);

    return output_shapes;
}
}  // namespace v6
}  // namespace op
}  // namespace ov
