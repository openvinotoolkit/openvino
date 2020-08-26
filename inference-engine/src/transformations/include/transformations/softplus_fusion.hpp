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

class TRANSFORMATIONS_API SoftPlusFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusFusion transformation replaces group of
 * operations: log(exp(x) + 1) to SoftPlus op.
 */
class ngraph::pass::SoftPlusFusion: public ngraph::pass::MatcherPass {
public:
    SoftPlusFusion();
};
