// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

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
class ngraph::pass::ReduceL2Decomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceL2Decomposition", "0");
    ReduceL2Decomposition();
};
