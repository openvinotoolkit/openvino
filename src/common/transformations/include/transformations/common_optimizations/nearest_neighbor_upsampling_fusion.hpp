// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NearestNeighborUpsamplingFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief NearestNeighborUpsamplingFusion transformation fuses subgraph that uses the simpler operations, as ShapeOf,
 *        StridedSlice, Concat, Reshape, Mul to calculate Interpolate with mode='nearest'.
 */
class ov::pass::NearestNeighborUpsamplingFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NearestNeighborUpsamplingFusion", "0");
    NearestNeighborUpsamplingFusion();
};
