// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NearestNeighborUpsamplingFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief NearestNeighborUpsamplingFusion transformation fuses subgraph that uses the simpler operations, as ShapeOf,
 *        StridedSlice, Concat, Reshape, Mul to calculate Interpolate with mode='nearest'.
 */
class ov::pass::NearestNeighborUpsamplingFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NearestNeighborUpsamplingFusion", "0");
    NearestNeighborUpsamplingFusion();
};

namespace ngraph {
namespace pass {
using ov::pass::NearestNeighborUpsamplingFusion;
}  // namespace pass
}  // namespace ngraph
