// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ClampFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ClampFusion transformation replaces following graph:
 * Maximum->Minimum to Clamp
 * Restrictions:
 * - one of the parameters to Maximum is a scalar constant
 * - one of the parameters to Minimum is a scalar constant
 */

class ngraph::pass::ClampFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ClampFusion();
};
