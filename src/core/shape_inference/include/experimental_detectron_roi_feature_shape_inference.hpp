// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dimension_util.hpp"
#include "openvino/op/experimental_detectron_roi_feature.hpp"
#include "utils.hpp"

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
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ExperimentalDetectronROIFeatureExtractor* op,
                                 const std::vector<TShape>& input_shapes) {
    using TDim = typename TRShape::value_type;
    using namespace ov::util;
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= 2);

    auto output_shapes = std::vector<TRShape>();
    output_shapes.reserve(2);

    const auto& rois_shape = input_shapes[0];
    const auto rois_shape_rank = rois_shape.rank();
    NODE_VALIDATION_CHECK(op, rois_shape_rank.compatible(2), "Input rois rank must be equal to 2.");

    if (rois_shape_rank.is_static()) {
        output_shapes.emplace_back(std::initializer_list<TDim>{rois_shape[0], TDim(dim::inf_bound)});
        output_shapes.emplace_back(std::initializer_list<TDim>{rois_shape[0], 4});
        auto& out_rois_shape = output_shapes[1];

        NODE_VALIDATION_CHECK(op,
                              TDim::merge(out_rois_shape[1], out_rois_shape[1], rois_shape[1]),
                              "The last dimension of the 'input_rois' input must be equal to 4. "
                              "Got: ",
                              rois_shape[1]);
    } else {
        output_shapes.emplace_back(std::initializer_list<TDim>{TDim(dim::inf_bound), TDim(dim::inf_bound)});
        output_shapes.emplace_back(std::initializer_list<TDim>{TDim(dim::inf_bound), 4});
    }

    auto& out_rois_feat_shape = output_shapes[0];
    out_rois_feat_shape.insert(out_rois_feat_shape.end(), 2, TDim(op->get_attrs().output_size));

    bool channels_intersection_initialized = false;
    for (size_t i = 1; i < input_shapes.size(); ++i) {
        const auto& layer_shape = input_shapes[i];
        const auto layer_rank = layer_shape.rank();
        NODE_VALIDATION_CHECK(op,
                              layer_rank.compatible(4),
                              "Rank of each element of the pyramid must be equal to 4. Got: ",
                              layer_rank);

        if (layer_rank.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  layer_shape[0].compatible(1),
                                  "The first dimension of each pyramid element must be equal to 1. Got: ",
                                  layer_shape[0]);

            if (channels_intersection_initialized) {
                NODE_VALIDATION_CHECK(op,
                                      TDim::merge(out_rois_feat_shape[1], out_rois_feat_shape[1], layer_shape[1]),
                                      "The number of channels must be the same for all layers of the pyramid.");
            } else {
                out_rois_feat_shape[1] = layer_shape[1];
                channels_intersection_initialized = true;
            }
        }
    }

    return output_shapes;
}
}  // namespace v6
}  // namespace op
}  // namespace ov
