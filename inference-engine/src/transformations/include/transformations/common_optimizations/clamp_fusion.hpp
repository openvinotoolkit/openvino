// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ClampFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ClampFusion transformation replaces following graph:
 * Maximum->Minimum to Clamp
 * Restrictions:
 * - one of the parameters to Maximum is a scalar constant
 * - one of the parameters to Minimum is a scalar constant
 */

class ov::pass::ClampFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ClampFusion();
};
