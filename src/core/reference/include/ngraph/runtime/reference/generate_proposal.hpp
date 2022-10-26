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
void generate_proposals(const std::vector<float>& im_info,
                        const std::vector<float>& anchors,
                        const std::vector<float>& deltas,
                        const std::vector<float>& scores,
                        const op::v9::GenerateProposals::Attributes& attrs,
                        const Shape& im_info_shape,
                        const Shape& anchors_shape,
                        const Shape& deltas_shape,
                        const Shape& scores_shape,
                        std::vector<float>& output_rois,
                        std::vector<float>& output_scores,
                        std::vector<int64_t>& num_rois);

void generate_proposals_postprocessing(void* prois,
                                       void* pscores,
                                       void* proi_num,
                                       const ngraph::element::Type& output_type,
                                       const ngraph::element::Type& roi_num_type,
                                       const std::vector<float>& output_rois,
                                       const std::vector<float>& output_scores,
                                       const std::vector<int64_t>& num_rois,
                                       const Shape& output_rois_shape,
                                       const Shape& output_scores_shape);

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
