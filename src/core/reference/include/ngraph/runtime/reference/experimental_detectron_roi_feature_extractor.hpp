// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void experimental_detectron_roi_feature_extractor(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<Shape>& input_shapes,
    const op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs,
    float* output_rois_features,
    float* output_rois);

void experimental_detectron_roi_feature_extractor_postprocessing(void* prois_features,
                                                                 void* prois,
                                                                 const ngraph::element::Type output_type,
                                                                 const std::vector<float>& output_roi_features,
                                                                 const std::vector<float>& output_rois,
                                                                 const Shape& output_roi_features_shape,
                                                                 const Shape& output_rois_shape);
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
