// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "nms_shape_inference.hpp"
#include "openvino/op/matrix_nms.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v8 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const MatrixNms* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using TDim = typename TRShape::value_type;
    using V = typename TDim::value_type;
    using namespace ov::util;

    nms::validate::boxes_shape(op, input_shapes);
    nms::validate::scores_shape(op, input_shapes);

    const auto& boxes_shape = input_shapes[0];
    const auto& scores_shape = input_shapes[1];
    const auto boxes_rank = boxes_shape.rank();
    const auto scores_rank = scores_shape.rank();

    auto output_shapes = std::vector<TRShape>{TRShape{TDim(dim::inf_bound), 6},
                                              TRShape{TDim(dim::inf_bound), 1},
                                              TRShape{TDim(dim::inf_bound)}};
    if (boxes_rank.is_static()) {
        const auto& nms_attrs = op->get_attrs();
        const auto nms_top_k = nms_attrs.nms_top_k;
        const auto keep_top_k = nms_attrs.keep_top_k;
        const auto background_class = nms_attrs.background_class;

        NODE_VALIDATION_CHECK(op, nms_top_k >= -1, "The 'nms_top_k' must be great or equal -1. Got:", nms_top_k);
        NODE_VALIDATION_CHECK(op, keep_top_k >= -1, "The 'keep_top_k' must be great or equal -1. Got:", keep_top_k);
        NODE_VALIDATION_CHECK(op,
                              background_class >= -1,
                              "The 'background_class' must be great or equal -1. Got:",
                              background_class);

        auto num_boxes = (nms_top_k > -1) ? TDim(std::min(boxes_shape[1].get_max_length(), static_cast<V>(nms_top_k)))
                                          : boxes_shape[1];

        if (scores_rank.is_static()) {
            nms::validate::num_batches(op, input_shapes);
            nms::validate::num_boxes(op, input_shapes);
            num_boxes *= scores_shape[1];
            if (keep_top_k > -1) {
                num_boxes = TDim(std::min(num_boxes.get_max_length(), static_cast<V>(keep_top_k)));
            }
            num_boxes *= scores_shape[0];
            const auto& selected_boxes =
                std::is_same<TRShape, PartialShape>::value ? TDim(0, num_boxes.get_max_length()) : std::move(num_boxes);

            std::for_each(output_shapes.begin(), output_shapes.begin() + 2, [&selected_boxes](TRShape& s) {
                s[0] = selected_boxes;
            });
        }
        nms::validate::boxes_last_dim(op, input_shapes);

        output_shapes[2][0] = boxes_shape[0];
    }

    return output_shapes;
}
}  // namespace v8
}  // namespace op
}  // namespace ov
