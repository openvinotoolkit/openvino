// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_topkrois.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v6 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(ExperimentalDetectronTopKROIs* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& input_rois_shape = input_shapes[0];
    const auto& rois_probs_shape = input_shapes[1];

    const auto input_rois_rank = input_rois_shape.rank();
    const auto rois_probs_rank = rois_probs_shape.rank();

    if (input_rois_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              input_rois_rank.get_length() == 2,
                              "The 'input_rois' input is expected to be a 2D. Got: ",
                              input_rois_rank);
        NODE_VALIDATION_CHECK(op,
                              input_rois_shape[1].compatible(4),
                              "The second dimension of 'input_rois' should be 4. Got: ",
                              input_rois_shape[1]);
    }

    NODE_VALIDATION_CHECK(op,
                          rois_probs_rank.compatible(1),
                          "The 'rois_probs' input is expected to be a 1D. Got: ",
                          rois_probs_rank);

    if (input_rois_rank.is_static() && rois_probs_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              input_rois_shape[0].compatible(rois_probs_shape[0]),
                              "Number of rois and number of probabilities should be equal. Got: ",
                              input_rois_shape[0],
                              " ",
                              rois_probs_shape[0]);
    }

    return {{static_cast<typename TShape::value_type>(op->get_max_rois()), 4}};
}
}  // namespace v6
}  // namespace op
}  // namespace ov
