// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/proposal.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace proposal {
template <class TOp, class TShape, class TRShape = result_shape_t<TShape>>
TRShape shape_infer_boxes(const TOp* op, const std::vector<TShape>& input_shapes) {
    using TDim = typename TRShape::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    const auto& class_probs_ps = input_shapes[0];
    const auto& bbox_deltas_ps = input_shapes[1];
    const auto& image_shape_ps = input_shapes[2];

    NODE_VALIDATION_CHECK(op,
                          class_probs_ps.rank().compatible(4),
                          "Proposal layer shape class_probs should be rank 4 compatible (",
                          class_probs_ps,
                          ").");

    NODE_VALIDATION_CHECK(op,
                          bbox_deltas_ps.rank().compatible(4),
                          "Proposal layer shape bbox_deltas should be rank 4 compatible (",
                          bbox_deltas_ps,
                          ").");

    if (image_shape_ps.rank().is_static()) {
        NODE_VALIDATION_CHECK(
            op,
            image_shape_ps.size() == 1 && (image_shape_ps[0].compatible(3) || image_shape_ps[0].compatible(4)),
            "Image_shape must be 1-D tensor and has got 3 or 4 elements (image_shape_shape[0]",
            image_shape_ps,
            ").");
    }

    const auto is_bbox_rank_dynamic = bbox_deltas_ps.rank().is_dynamic();

    TRShape proposed_boxes_shape;
    proposed_boxes_shape.reserve(2);

    if (class_probs_ps.rank().is_static()) {
        proposed_boxes_shape.push_back(class_probs_ps[0]);

        // check anchor count and batch number consistency
        NODE_VALIDATION_CHECK(op,
                              is_bbox_rank_dynamic || bbox_deltas_ps[1].compatible(class_probs_ps[1] * 2),
                              "Anchor number inconsistent between class_probs (",
                              class_probs_ps[1] * 2,
                              "), and bbox_deltas (",
                              bbox_deltas_ps[1],
                              ").");

    } else {
        proposed_boxes_shape.emplace_back(ov::util::dim::inf_bound);
    }

    NODE_VALIDATION_CHECK(
        op,
        is_bbox_rank_dynamic || TDim::merge(proposed_boxes_shape[0], proposed_boxes_shape[0], bbox_deltas_ps[0]),
        "Batch size inconsistent between class_probs (",
        class_probs_ps[0],
        ") and bbox deltas (",
        bbox_deltas_ps[0],
        ").");

    proposed_boxes_shape[0] *= op->get_attrs().post_nms_topn;
    proposed_boxes_shape.emplace_back(5);
    return proposed_boxes_shape;
}
}  // namespace proposal

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Proposal* op, const std::vector<TShape>& input_shapes) {
    return {ov::op::proposal::shape_infer_boxes(op, input_shapes)};
}
}  // namespace v0
}  // namespace op
}  // namespace ov

namespace ov {
namespace op {
namespace v4 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Proposal* op, const std::vector<TShape>& input_shapes) {
    auto output_shapes = std::vector<TRShape>(2, ov::op::proposal::shape_infer_boxes(op, input_shapes));
    output_shapes[1].resize(1);
    return output_shapes;
}

}  // namespace v4
}  // namespace op
}  // namespace ov
