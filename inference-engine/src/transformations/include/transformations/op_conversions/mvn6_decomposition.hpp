// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API MVN6Decomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MVN6Decomposition transformation into sub-graph x - ReduceMean(x, axes) if normalize_variance is false and
 * into sub-graph (x - ReduceMean(x, axes)) / Sqrt(ReduceSum((x - ReduceMean(x, axes)) ^ 2)) if normalize_variance is true.
 */
class ngraph::pass::MVN6Decomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MVN6Decomposition();
};
