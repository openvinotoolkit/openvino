// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/op/non_max_suppression.hpp>
#include <vector>

using namespace ngraph;

namespace ov {
namespace op {
namespace v9 {

template <class T>
void shape_infer(const NonMaxSuppression* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 3);

    const auto& boxes_ps = input_shapes[0];
    const auto& scores_ps = input_shapes[1];

    if (!boxes_ps.is_dynamic() && !scores_ps.is_dynamic()) {
        NODE_VALIDATION_CHECK(op,
                              boxes_ps.rank().is_static() && boxes_ps.rank().get_length() == 3,
                              "Expected a 3D tensor for the 'boxes' input. Got: ",
                              boxes_ps);

        NODE_VALIDATION_CHECK(op,
                              scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3,
                              "Expected a 3D tensor for the 'scores' input. Got: ",
                              scores_ps);

        const auto num_batches_boxes = boxes_ps[0];
        const auto num_batches_scores = scores_ps[0];
        NODE_VALIDATION_CHECK(op,
                              num_batches_boxes.same_scheme(num_batches_scores),
                              "The first dimension of both 'boxes' and 'scores' must match. Boxes: ",
                              num_batches_boxes,
                              "; Scores: ",
                              num_batches_scores);

        const auto num_boxes_boxes = boxes_ps[1];
        const auto num_boxes_scores = scores_ps[2];
        NODE_VALIDATION_CHECK(op,
                              num_boxes_boxes.same_scheme(num_boxes_scores),
                              "'boxes' and 'scores' input shapes must match at the second and third "
                              "dimension respectively. Boxes: ",
                              num_boxes_boxes,
                              "; Scores: ",
                              num_boxes_scores);

        NODE_VALIDATION_CHECK(op,
                              boxes_ps[2].is_static() && boxes_ps[2].get_length() == 4u,
                              "The last dimension of the 'boxes' input must be equal to 4. Got:",
                              boxes_ps[2]);
    }

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape out_shape = {Dimension::dynamic(), 3};
    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static() && op->get_input_size() > 2) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static() &&
            has_and_set_equal_bounds(op->input_value(2))) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            const auto max_output_boxes_per_class = op->max_boxes_output_from_input();

            out_shape[0] =
                Dimension(0, std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length());
        }
    }
    output_shapes[0] = out_shape;
    output_shapes[1] = out_shape;
    output_shapes[2] = ov::Shape{1};
}

}  // namespace v9
}  // namespace op
}  // namespace ov
