// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API AddFakeQuantizeFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief AddFakeQuantizeFusion transformation replaces following graph:
 * Add->FakeQuantize to a single FakeQuantize
 * Restrictions:
 * - second input to Add is a Constant
 */
class ngraph::pass::AddFakeQuantizeFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddFakeQuantizeFusion();
};
