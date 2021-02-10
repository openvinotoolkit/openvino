// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReduceL2Decomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Decomposes ReduceL2 into sqrt(ReduceSum(x * x)).
 */
class ngraph::pass::ReduceL2Decomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReduceL2Decomposition();
};
