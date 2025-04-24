// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/op/experimental_detectron_roi_feature.hpp"

namespace ov {
namespace reference {
void experimental_detectron_roi_feature_extractor(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<Shape>& input_shapes,
    const op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs,
    float* output_rois_features,
    float* output_rois);

void experimental_detectron_roi_feature_extractor_postprocessing(void* prois_features,
                                                                 void* prois,
                                                                 const element::Type output_type,
                                                                 const std::vector<float>& output_roi_features,
                                                                 const std::vector<float>& output_rois,
                                                                 const Shape& output_roi_features_shape,
                                                                 const Shape& output_rois_shape);
}  // namespace reference
}  // namespace ov
