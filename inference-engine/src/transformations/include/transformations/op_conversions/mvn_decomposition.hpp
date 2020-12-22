// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API MVNDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNDecomposition transformation into sub-graph (x - ReduceMean(x, axes)) / Sqrt(ReduceSum((x - ReduceMean(x, axes)) ^ 2)).
 */
class ngraph::pass::MVNDecomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNDecomposition();
};
