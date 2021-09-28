// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/experimental_detectron_roi_feature.hpp>

namespace ov {
namespace op {
namespace v6 {
// by definition:
// inputs:
//          1.    [number_of_ROIs, 4]
//          2..L  [1, number_of_channels, layer_size[l], layer_size[l]]
// outputs:
//          1.  out_shape = [number_of_ROIs, number_of_channels, output_size, output_size]
//          2.  out_rois_shape = [number_of_ROIs, 4]
template <class T>
void shape_infer(ExperimentalDetectronROIFeatureExtractor* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    using DimensionType = typename std::decay<decltype(output_shapes[0][0])>::type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() >= 2 && output_shapes.size() == 2);

    const auto& rois_shape = input_shapes[0];
    auto& out_shape = output_shapes[0];
    auto& out_rois_shape = output_shapes[1];

    // all dimensions is initialized by-default as dynamic
    out_shape.resize(4);
    out_rois_shape.resize(2);

    // infer static dimensions
    out_shape[2] = op->m_attrs.output_size;
    out_shape[3] = op->m_attrs.output_size;
    out_rois_shape[1] = 4;

    // infer number_of_ROIs (which may be dynamic/static)
    if (rois_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, rois_shape.rank().get_length() == 2, "Input rois rank must be equal to 2.");

        auto input_rois_last_dim_intersection_with_4 = rois_shape[1] & DimensionType(4);
        NODE_VALIDATION_CHECK(op,
                              !input_rois_last_dim_intersection_with_4.get_interval().empty(),
                              "The last dimension of the 'input_rois' input must be equal to 4. "
                              "Got: ",
                              rois_shape[1]);

        out_shape[0] = rois_shape[0];
        out_rois_shape[0] = rois_shape[0];
    }

    // infer number_of_channels;
    // by definition, all shapes starting from input 2 must have same number_of_channels
    DimensionType channels_intersection;
    for (size_t i = 1; i < input_shapes.size(); i++) {
        auto& current_shape = input_shapes[i];
        auto current_rank = current_shape.rank();

        if (current_rank.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  current_rank.get_length() == 4,
                                  "Rank of each element of the pyramid must be equal to 4. Got: ",
                                  current_rank);

            auto first_dim_intersection_with_1 = current_shape[0] & DimensionType(1);
            NODE_VALIDATION_CHECK(op,
                                  !first_dim_intersection_with_1.get_interval().empty(),
                                  "The first dimension of each pyramid element must be equal to 1. "
                                  "Got: ",
                                  current_shape[0]);

            channels_intersection &= current_shape[1];
        }
    }

    NODE_VALIDATION_CHECK(op,
                          !channels_intersection.get_interval().empty(),
                          "The number of channels must be the same for all layers of the pyramid.");

    out_shape[1] = channels_intersection;
}
}  // namespace v6
}  // namespace op
}  // namespace ov
