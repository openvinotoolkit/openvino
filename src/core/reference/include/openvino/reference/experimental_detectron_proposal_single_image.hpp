// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

namespace ov {
namespace reference {
void experimental_detectron_proposals_single_image(
    const float* im_info,
    const float* anchors,
    const float* deltas,
    const float* scores,
    const op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes& attrs,
    const Shape& im_info_shape,
    const Shape& anchors_shape,
    const Shape& deltas_shape,
    const Shape& scores_shape,
    float* output_rois,
    float* output_scores);

void experimental_detectron_proposals_single_image_postprocessing(void* prois,
                                                                  void* pscores,
                                                                  const element::Type output_type,
                                                                  const std::vector<float>& output_rois,
                                                                  const std::vector<float>& output_scores,
                                                                  const Shape& output_rois_shape,
                                                                  const Shape& output_scores_shape);
}  // namespace reference
}  // namespace ov
