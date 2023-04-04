// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/op/non_max_suppression.hpp>
#include <vector>

#include "utils.hpp"

using namespace ngraph;

namespace ov {
namespace op {
namespace v9 {

template <class T>
void shape_infer(const NonMaxSuppression* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 bool static_output = false,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    // this shape_infer differs from all the other - it is used in GPU during compile-time and infer-time in custom code
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
    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        bool gpu_wa =
            static_output && (!num_boxes_boxes.is_static() || !scores_ps[0].is_static() || !scores_ps[1].is_static());
        if (!gpu_wa && num_boxes_boxes.get_max_length() != -1 && scores_ps[0].get_max_length() != -1 &&
            scores_ps[1].get_max_length() != -1) {
            const auto num_boxes = num_boxes_boxes.get_max_length();
            const auto num_classes = scores_ps[1].get_max_length();
            std::vector<int64_t> max_output_boxes_per_class_as_vals;
            if ((op->get_input_size() > 2 || constant_data.count(2)) &&
                get_data_as_int64<T>(2, op, max_output_boxes_per_class_as_vals, constant_data)) {
                int64_t max_output_boxes_per_class = max_output_boxes_per_class_as_vals[0];
                out_shape[0] = static_output ? std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                                                   scores_ps[0].get_max_length()
                                             : Dimension(0,
                                                         std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                                                             scores_ps[0].get_max_length());
            }
        }
    }
    output_shapes[0] = out_shape;
    output_shapes[1] = out_shape;
    output_shapes[2] = ov::Shape{1};
}

}  // namespace v9
}  // namespace op
}  // namespace ov
