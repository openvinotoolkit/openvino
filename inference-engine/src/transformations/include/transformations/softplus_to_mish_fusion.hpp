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

class TRANSFORMATIONS_API SoftPlusToMishFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftPlusToMishFusion transformation replaces group of
 * operations: x * tanh(softplus(x)) to Mish op.
 */
class ngraph::pass::SoftPlusToMishFusion: public ngraph::pass::MatcherPass {
public:
    SoftPlusToMishFusion();
};
