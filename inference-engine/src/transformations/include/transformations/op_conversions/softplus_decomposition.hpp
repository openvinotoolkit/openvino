// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SoftPlusDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusDecomposition transformation replaces SoftPlus op to
 * group of operations: log(exp(x) + 1).
 */
class ngraph::pass::SoftPlusDecomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SoftPlusDecomposition();
};
