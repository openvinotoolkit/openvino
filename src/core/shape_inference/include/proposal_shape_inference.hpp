// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/proposal.hpp>

namespace ov {
namespace op {
namespace v0 {
template <class OpType, class ShapeType>
void infer_prop_shape(const OpType* op,
                      const std::vector<ShapeType>& input_shapes,
                      std::vector<ShapeType>& output_shapes) {
    using DimType = typename std::iterator_traits<typename ShapeType::iterator>::value_type;
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

    NODE_VALIDATION_CHECK(op,
                          image_shape_ps.rank().compatible(1),
                          "Proposal layer shape image_shape should be rank 1 compatible (",
                          image_shape_ps,
                          ").");
    if (bbox_deltas_ps.rank().is_static() && class_probs_ps.rank().is_static()) {
        // check anchor count and batch number consistency
        NODE_VALIDATION_CHECK(op,
                              bbox_deltas_ps[1].compatible(class_probs_ps[1] * 2),
                              "Anchor number inconsistent between class_probs (",
                              class_probs_ps[1] * 2,
                              "), and bbox_deltas (",
                              bbox_deltas_ps[1],
                              ").");

        NODE_VALIDATION_CHECK(op,
                              class_probs_ps[0].compatible(bbox_deltas_ps[0]),
                              "Batch size inconsistent between class_probs (",
                              class_probs_ps[0],
                              ") and bbox deltas (",
                              bbox_deltas_ps[0],
                              ").");
    }

    if (image_shape_ps.is_static()) {
        const auto image_shape_elem = image_shape_ps[0].get_length();
        NODE_VALIDATION_CHECK(op,
                              image_shape_elem >= 3 && image_shape_elem <= 4,
                              "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
                              image_shape_ps[0],
                              ").");
    }

    auto out_dim = DimType{};

    if (class_probs_ps.rank().is_static() && bbox_deltas_ps.rank().is_static()) {
        OPENVINO_ASSERT(DimType::merge(out_dim, class_probs_ps[0], bbox_deltas_ps[0]));
    } else if (class_probs_ps.rank().is_static()) {
        out_dim = class_probs_ps[0];
    } else if (bbox_deltas_ps.rank().is_static()) {
        out_dim = bbox_deltas_ps[0];
    } else {
        out_dim = Dimension::dynamic();
    }

    auto& proposed_boxes_shape = output_shapes[0];
    proposed_boxes_shape.resize(2);
    proposed_boxes_shape[0] = out_dim * op->get_attrs().post_nms_topn;
    proposed_boxes_shape[1] = 5;
}
template <class T>
void shape_infer(const ov::op::v0::Proposal* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    ov::op::v0::infer_prop_shape(op, input_shapes, output_shapes);
}

}  // namespace v0
}  // namespace op
}  // namespace ov

namespace ov {
namespace op {
namespace v4 {

template <class T>
void shape_infer(const ov::op::v4::Proposal* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 2);

    ov::op::v0::infer_prop_shape(op, input_shapes, output_shapes);
    const auto& proposals_ps = output_shapes[0];
    auto& out_ps = output_shapes[1];
    out_ps = T{proposals_ps[0]};
}

}  // namespace v4
}  // namespace op
}  // namespace ov
