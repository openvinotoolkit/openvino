// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/experimental_detectron_topkrois.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(ExperimentalDetectronTopKROIs* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);

    const auto input_rois_shape = input_shapes[0];
    const auto rois_probs_shape = input_shapes[1];

    if (input_rois_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              input_rois_shape.rank().get_length() == 2,
                              "The 'input_rois' input is expected to be a 2D. Got: ",
                              input_rois_shape);
        NODE_VALIDATION_CHECK(op,
                              input_rois_shape[1].compatible(4),
                              "The second dimension of 'input_rois' should be 4. Got: ",
                              input_rois_shape[1]);
    }
    NODE_VALIDATION_CHECK(op,
                          rois_probs_shape.rank().compatible(1),
                          "The 'rois_probs' input is expected to be a 1D. Got: ",
                          rois_probs_shape);

    if (input_rois_shape.rank().is_static() && rois_probs_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              input_rois_shape[0].compatible(rois_probs_shape[0]),
                              "Number of rois and number of probabilities should be equal. Got: ",
                              input_rois_shape[0],
                              rois_probs_shape[0]);
    }

    auto& output_shape = output_shapes[0];
    auto max_rois = op->m_max_rois;

    output_shape.resize(2);
    output_shape[0] = max_rois;
    output_shape[1] = 4;
}

}  // namespace v6
}  // namespace op
}  // namespace ov
