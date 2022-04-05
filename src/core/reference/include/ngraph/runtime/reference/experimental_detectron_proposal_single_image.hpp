// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
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
                                                                  const ngraph::element::Type output_type,
                                                                  const std::vector<float>& output_rois,
                                                                  const std::vector<float>& output_scores,
                                                                  const Shape& output_rois_shape,
                                                                  const Shape& output_scores_shape);
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
