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

class TRANSFORMATIONS_API ReduceL1Decomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Decomposes ReduceL1 into ReduceSum(abs(x)).
 */
class ngraph::pass::ReduceL1Decomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReduceL1Decomposition();
};
