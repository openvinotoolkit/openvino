// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReduceL2Decomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Decomposes ReduceL2 into sqrt(ReduceSum(x * x)).
 */
class ov::pass::ReduceL2Decomposition: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReduceL2Decomposition();
};
