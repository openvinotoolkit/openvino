// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/multiclass_nms.hpp>
#include <openvino/op/util/multiclass_nms_base.hpp>
#include <vector>

namespace ov {
namespace op {
namespace util {

template <class T>
void shape_infer(const ov::op::util::MulticlassNmsBase* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 bool static_output = false,
                 bool ignore_bg_class = false) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3) && output_shapes.size() == 3);

    const auto& boxes_ps = input_shapes[0];
    const auto& scores_ps = input_shapes[1];

    const auto& nms_attrs = op->get_attrs();
    const auto nms_top_k = nms_attrs.nms_top_k;
    const auto keep_top_k = nms_attrs.keep_top_k;

    // validate rank of each input
    if (boxes_ps.rank().is_dynamic() || scores_ps.rank().is_dynamic()) {
        return;
    }

    if (op->get_input_size() == 3) {
        NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);
        const auto& roisnum_ps = input_shapes[2];
        if (roisnum_ps.rank().is_dynamic()) {
            return;
        }
    }

    // validate shape of each input
    NODE_VALIDATION_CHECK(op,
                          boxes_ps.rank().is_static() && boxes_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'boxes' input. Got: ",
                          boxes_ps);

    NODE_VALIDATION_CHECK(op,
                          boxes_ps[2].is_static() && boxes_ps[2].get_length() == 4,
                          "The third dimension of the 'boxes' must be 4. Got: ",
                          boxes_ps[2]);

    if (ov::is_type<ov::op::v8::MulticlassNms>(op)) {
        NODE_VALIDATION_CHECK(op,
                              scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3,
                              "Expected a 3D tensor for the 'scores' input. Got: ",
                              scores_ps);
    } else {
        NODE_VALIDATION_CHECK(
            op,
            scores_ps.rank().is_static() && (scores_ps.rank().get_length() == 3 || scores_ps.rank().get_length() == 2),
            "Expected a 2D or 3D tensor for the 'scores' input. Got: ",
            scores_ps);
    }

    if (op->get_input_size() == 3) {
        const auto& roisnum_ps = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              roisnum_ps.rank().is_static() && roisnum_ps.rank().get_length() == 1,
                              "Expected a 1D tensor for the 'roisnum' input. Got: ",
                              roisnum_ps);
    }

    // validate compatibility of input shapes
    if (scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3) {  // if scores shape (N, C, M)
        const auto num_batches_boxes = boxes_ps[0];
        const auto num_batches_scores = scores_ps[0];

        NODE_VALIDATION_CHECK(op,
                              num_batches_boxes.compatible(num_batches_scores),
                              "The first dimension of both 'boxes' and 'scores' must match. Boxes: ",
                              num_batches_boxes,
                              "; Scores: ",
                              num_batches_scores);

        const auto num_boxes_boxes = boxes_ps[1];
        const auto num_boxes_scores = scores_ps[2];
        NODE_VALIDATION_CHECK(op,
                              num_boxes_boxes.compatible(num_boxes_scores),
                              "'boxes' and 'scores' input shapes must match at the second and third "
                              "dimension respectively. Boxes: ",
                              num_boxes_boxes,
                              "; Scores: ",
                              num_boxes_scores);
    }

    if (scores_ps.rank().is_static() && scores_ps.rank().get_length() == 2) {  // if scores shape (C, M)
        NODE_VALIDATION_CHECK(op,
                              op->get_input_size() == 3,
                              "Expected the 'roisnum' input when the input 'scores' is a 2D tensor.");

        const auto num_classes_boxes = boxes_ps[0];
        const auto num_classes_scores = scores_ps[0];
        NODE_VALIDATION_CHECK(op,
                              num_classes_boxes.compatible(num_classes_scores),
                              "'boxes' and 'scores' input shapes must match. Boxes: ",
                              num_classes_boxes,
                              "; Scores: ",
                              num_classes_scores);

        const auto num_boxes_boxes = boxes_ps[1];
        const auto num_boxes_scores = scores_ps[1];
        NODE_VALIDATION_CHECK(op,
                              num_boxes_boxes.compatible(num_boxes_scores),
                              "'boxes' and 'scores' input shapes must match. Boxes: ",
                              num_boxes_boxes,
                              "; Scores: ",
                              num_boxes_scores);
    }

    /* rank of inputs have been static since here. */
    auto _ready_infer = [&]() {
        if (boxes_ps.rank().is_dynamic() || scores_ps.rank().is_dynamic()) {
            return false;
        }
        const bool shared = (scores_ps.rank().get_length() == 3);
        if (shared) {
            return boxes_ps[1].is_static() && scores_ps[1].is_static() && scores_ps[0].is_static();
        } else {
            const auto& roisnum_ps = input_shapes[2];
            if (roisnum_ps.rank().is_dynamic()) {
                return false;
            }
            return boxes_ps[1].is_static() && boxes_ps[0].is_static() && roisnum_ps[0].is_static();
        }
    };

    // Here output 0 and output 1 is not the real dimension of output.
    // It will be rewritten in the computing runtime.
    // But we still need it here for static shape only backends.
    auto first_dim_shape = Dimension::dynamic();
    if (_ready_infer()) {
        const bool shared = (scores_ps.rank().get_length() == 3);
        ov::PartialShape roisnum_ps;
        if (!shared) {
            roisnum_ps = input_shapes[2];
        }

        const auto num_boxes = shared ? boxes_ps[1].get_length() : boxes_ps[1].get_length();
        auto num_classes = shared ? scores_ps[1].get_length() : boxes_ps[0].get_length();
        auto num_images = shared ? scores_ps[0].get_length() : roisnum_ps[0].get_length();

        if (ignore_bg_class) {
            if (nms_attrs.background_class >= 0 && nms_attrs.background_class < num_classes) {
                num_classes = std::max(int64_t{1}, num_classes - 1);
            }
        }

        int64_t max_output_boxes_per_class = 0;
        if (nms_top_k >= 0)
            max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
        else
            max_output_boxes_per_class = num_boxes;

        auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
        if (keep_top_k >= 0)
            max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

        first_dim_shape = static_output ? max_output_boxes_per_batch * num_images
                                        : Dimension(0, max_output_boxes_per_batch * num_images);
    }

    // 'selected_outputs' have the following format:
    //      [number of selected boxes, [class_id, box_score, xmin, ymin, xmax, ymax]]
    output_shapes[0] = {first_dim_shape, 6};
    // 'selected_indices' have the following format:
    //      [number of selected boxes, ]
    output_shapes[1] = {first_dim_shape, 1};
    // 'selected_num' have the following format:
    //      [num_batches, ]
    if (op->get_input_size() == 3) {
        const auto& roisnum_ps = input_shapes[2];
        if (roisnum_ps.rank().is_static() && roisnum_ps.rank().get_length() > 0) {
            output_shapes[2] = {roisnum_ps[0]};
        } else {
            output_shapes[2] = {Dimension::dynamic()};
        }
    } else {  // shared
        if (boxes_ps.rank().is_static() && boxes_ps.rank().get_length() > 0) {
            output_shapes[2] = {boxes_ps[0]};
        } else {
            output_shapes[2] = {Dimension::dynamic()};
        }
    }
}

}  // namespace util
}  // namespace op
}  // namespace ov
