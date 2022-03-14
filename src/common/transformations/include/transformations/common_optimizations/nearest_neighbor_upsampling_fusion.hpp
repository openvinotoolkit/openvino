// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API NearestNeighborUpsamplingFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief NearestNeighborUpsamplingFusion transformation fuses subgraph that uses the simpler operations, as ShapeOf,
 *        StridedSlice, Concat, Reshape, Mul to calculate Interpolate with mode='nearest'.
 */
class ngraph::pass::NearestNeighborUpsamplingFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("NearestNeighborUpsamplingFusion", "0");
    NearestNeighborUpsamplingFusion();
};
