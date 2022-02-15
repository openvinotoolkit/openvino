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
void generate_proposals_single_image(
    const float* im_info,
    const float* anchors,
    const float* deltas,
    const float* scores,
    const op::v6::GenerateProposalsSingleImage::Attributes& attrs,
    const Shape& im_info_shape,
    const Shape& anchors_shape,
    const Shape& deltas_shape,
    const Shape& scores_shape,
    float* output_rois,
    float* output_scores);

void generate_proposals_single_image_postprocessing(void* prois,
                                                                  void* pscores,
                                                                  const ngraph::element::Type output_type,
                                                                  const std::vector<float>& output_rois,
                                                                  const std::vector<float>& output_scores,
                                                                  const Shape& output_rois_shape,
                                                                  const Shape& output_scores_shape);

void generate_proposals_single_image_v9(
    const float* im_info,
    const float* anchors,
    const float* deltas,
    const float* scores,
    const op::v9::GenerateProposalsSingleImage::Attributes& attrs,
    const Shape& im_info_shape,
    const Shape& anchors_shape,
    const Shape& deltas_shape,
    const Shape& scores_shape,
    std::vector<float>& output_rois,
    std::vector<float>& output_scores,
    std::vector<int64_t>& output_num);

void generate_proposals_single_image_postprocessing_v9(void* prois,
                                                                     void* pscores,
                                                                     const ngraph::element::Type output_type,
                                                                     const std::vector<float>& output_rois,
                                                                     const std::vector<float>& output_scores,
                                                                     const Shape& output_rois_shape,
                                                                     const Shape& output_scores_shape);

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
