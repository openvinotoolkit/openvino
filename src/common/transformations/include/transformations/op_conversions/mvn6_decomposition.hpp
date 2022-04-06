// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MVN6Decomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MVN6Decomposition transformation into sub-graph x - ReduceMean(x, axes) if normalize_variance is false and
 * into sub-graph (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) if normalize_variance is
 * true.
 */
class ngraph::pass::MVN6Decomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MVN6Decomposition", "0");
    MVN6Decomposition();
};
