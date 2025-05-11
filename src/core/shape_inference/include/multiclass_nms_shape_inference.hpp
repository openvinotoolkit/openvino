// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "nms_shape_inference.hpp"
#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/util/multiclass_nms_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace multiclass_nms {
namespace validate {
template <class TShape>
void scores_shape(const Node* const op, const std::vector<TShape>& input_shapes) {
    const auto scores_rank = input_shapes[1].rank();
    NODE_SHAPE_INFER_CHECK(op, input_shapes, scores_rank.compatible(2), "Expected a 2D tensor for the 'scores' input");
}
template <class TShape>
void rois_num_shape(const Node* const op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes[2].rank().compatible(1),
                           "Expected a 1D tensor for the 'roisnum' input");
}

template <class TShape>
void num_boxes(const Node* const op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes[0][1].compatible(input_shapes[1][1]),
                           "'boxes' and 'scores' input shapes must match at the second dimension respectively");
}
}  // namespace validate
}  // namespace multiclass_nms

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::MulticlassNmsBase* op,
                                 const std::vector<TShape>& input_shapes,
                                 const bool static_output = !std::is_same<PartialShape, TShape>::value,
                                 const bool ignore_bg_class = false) {
    const auto inputs_size = input_shapes.size();
    const auto has_rois_num = inputs_size == 3;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || has_rois_num));

    using TDim = typename TRShape::value_type;
    using V = typename TDim::value_type;
    using namespace ov::util;

    nms::validate::boxes_shape(op, input_shapes);
    if (has_rois_num) {
        multiclass_nms::validate::scores_shape(op, input_shapes);
        multiclass_nms::validate::rois_num_shape(op, input_shapes);
    } else {
        nms::validate::scores_shape(op, input_shapes);
    }

    auto output_shapes = std::vector<TRShape>{TRShape{TDim(dim::inf_bound), 6},
                                              TRShape{TDim(dim::inf_bound), 1},
                                              TRShape{TDim(dim::inf_bound)}};

    const auto& boxes_shape = input_shapes[0];
    const auto& scores_shape = input_shapes[1];
    const auto& rois_num_shape = has_rois_num ? input_shapes[2] : PartialShape::dynamic();

    if (boxes_shape.rank().is_static()) {
        const auto scores_rank = scores_shape.rank();
        nms::validate::num_batches(op, input_shapes);
        nms::validate::boxes_last_dim(op, input_shapes);

        bool can_infer;
        if (has_rois_num && scores_rank.is_static()) {
            multiclass_nms::validate::num_boxes(op, input_shapes);
            can_infer = rois_num_shape.rank().is_static();
        } else if (!has_rois_num && scores_rank.is_static()) {
            nms::validate::num_boxes(op, input_shapes);
            can_infer = true;
        } else {
            can_infer = false;
        }

        if (can_infer) {
            const auto& nms_attrs = op->get_attrs();
            const auto nms_top_k = nms_attrs.nms_top_k;
            const auto keep_top_k = nms_attrs.keep_top_k;
            const auto background_class = nms_attrs.background_class;

            const auto& num_classes = has_rois_num ? boxes_shape[0] : scores_shape[1];
            const auto& num_images = has_rois_num ? rois_num_shape[0] : scores_shape[0];

            auto& selected_boxes = output_shapes[0][0];
            selected_boxes =
                (nms_top_k > -1) ? TDim(std::min<V>(boxes_shape[1].get_max_length(), nms_top_k)) : boxes_shape[1];

            if (ignore_bg_class && (background_class > -1) && (background_class < num_classes.get_max_length())) {
                selected_boxes *= std::max<V>(1, num_classes.get_max_length() - 1);
            } else {
                selected_boxes *= num_classes;
            }

            if (keep_top_k > -1 && (keep_top_k < selected_boxes.get_max_length())) {
                selected_boxes = TDim(keep_top_k);
            }

            selected_boxes *= num_images;
            if (std::is_same<PartialShape, TShape>::value) {
                selected_boxes =
                    static_output ? TDim(selected_boxes.get_max_length()) : TDim(0, selected_boxes.get_max_length());
            }
            output_shapes[1][0] = selected_boxes;
            output_shapes[2][0] = has_rois_num ? rois_num_shape[0] : boxes_shape[0];
        }
    }

    return output_shapes;
}
}  // namespace op
}  // namespace ov
