// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include "low_precision/lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PullReshapeThroughDequantization;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief PullReshapeThroughDequantization propagates dequantization operations through Reshape operations.
 * The transformation is used on constant subgraph weights to prepare a model for the next low precision transformations.
 *
 * For more details about the transformation, refer to
 * [PullReshapeThroughDequantization](@ref openvino_docs_OV_UG_lpt_PullReshapeThroughDequantization) page
 * in the OpenVINO Developer Guide.
 */
class ov::pass::low_precision::PullReshapeThroughDequantization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PullReshapeThroughDequantization", "0");
    PullReshapeThroughDequantization(const std::vector<ov::element::Type>& inputPrecisions = {});
};
