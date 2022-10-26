// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

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

class ngraph::pass::ClampFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ClampFusion", "0");
    ClampFusion();
};
