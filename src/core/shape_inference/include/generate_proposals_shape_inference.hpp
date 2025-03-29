// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/generate_proposals.hpp>

#include "utils.hpp"

namespace ov {
namespace op {

namespace v9 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const GenerateProposals* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& im_info_shape = input_shapes[0];
    const auto& anchors_shape = input_shapes[1];
    const auto& deltas_shape = input_shapes[2];
    const auto& scores_shape = input_shapes[3];
    const auto im_info_shape_rank = im_info_shape.rank();
    auto num_batches = Dimension::dynamic();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           im_info_shape_rank.compatible(2),
                           "The 'input_im_info' input is expected to be a 2D. Got: ",
                           im_info_shape);

    if (im_info_shape_rank.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               (im_info_shape[1].compatible(3) || im_info_shape[1].compatible(4)),
                               "The 'input_im_info' shape[1] is expected to be a compatible with [3] or [4]. Got: ",
                               im_info_shape);
        Dimension::merge(num_batches, im_info_shape[0], num_batches);
    }

    const auto anchors_shape_rank = anchors_shape.rank();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           anchors_shape_rank.compatible(4),
                           "The 'input_anchors' input is expected to be a 4D. Got: ",
                           anchors_shape);
    if (anchors_shape_rank.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               anchors_shape[3].compatible(4),
                               "The fourth dimension of 'input_anchors' should be compatible with 4. Got: ",
                               anchors_shape[3]);
    }
    const auto deltas_shape_rank = deltas_shape.rank();
    const auto scores_shape_rank = scores_shape.rank();

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           deltas_shape_rank.compatible(4),
                           "The 'input_deltas' input is expected to be a 4D. Got: ",
                           deltas_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           scores_shape_rank.compatible(4),
                           "The 'input_scores' input is expected to be a 4D. Got: ",
                           scores_shape);
    if (deltas_shape_rank.is_static())
        Dimension::merge(num_batches, deltas_shape[0], num_batches);
    if (scores_shape_rank.is_static())
        Dimension::merge(num_batches, scores_shape[0], num_batches);

    if (deltas_shape_rank.is_static() && scores_shape_rank.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               deltas_shape[0].compatible(scores_shape[0]),
                               "Batch for inputs 'input_deltas' and 'input_scores' should be "
                               "equal. Got: ",
                               deltas_shape[0],
                               scores_shape[0]);

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               deltas_shape[1].compatible(scores_shape[1] * 4),
                               "Anchor number for inputs 'input_deltas' and 'input_scores' should be "
                               "equal. Got: ",
                               deltas_shape[1] / 4,
                               scores_shape[1]);

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               deltas_shape[2].compatible(scores_shape[2]),
                               "Heights for inputs 'input_deltas' and 'input_scores' should be "
                               "equal. Got: ",
                               deltas_shape[2],
                               scores_shape[2]);

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               deltas_shape[3].compatible(scores_shape[3]),
                               "Width for inputs 'input_deltas' and 'input_scores' should be "
                               "equal. Got: ",
                               deltas_shape[3],
                               scores_shape[3]);

        if (im_info_shape_rank.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   deltas_shape[0].compatible(im_info_shape[0]),
                                   "Batch for inputs 'im_info' and 'input_deltas' should be "
                                   "equal. Got: ",
                                   deltas_shape[0],
                                   im_info_shape[0]);
        }
    }

    if (scores_shape_rank.is_static() && anchors_shape_rank.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               anchors_shape[0].compatible(scores_shape[2]),
                               "Heights for inputs 'input_anchors' and 'input_scores' should be "
                               "equal. Got: ",
                               anchors_shape[0],
                               scores_shape[2]);

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               anchors_shape[1].compatible(scores_shape[3]),
                               "Width for inputs 'input_anchors' and 'input_scores' should be "
                               "equal. Got: ",
                               anchors_shape[1],
                               scores_shape[3]);

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               anchors_shape[2].compatible(scores_shape[1]),
                               "Anchor number for inputs 'input_anchors' and 'input_scores' should be "
                               "equal. Got: ",
                               anchors_shape[2],
                               scores_shape[1]);
    }

    auto num_rois = Dimension(0, (num_batches * op->get_attrs().post_nms_count).get_max_length());
    return {TRShape{num_rois, 4}, TRShape{num_rois}, TRShape{std::move(num_batches)}};
}

}  // namespace v9
}  // namespace op
}  // namespace ov
