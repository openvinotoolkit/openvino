// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API FakeQuantizeReshapeFusion;

} // namespace pass
} // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief This transformation looks for a FQ + Reshape pair in the graph and moves
 * the Reshape operation above the FQ node. Shapes of limit inputs are updated
 * following FQ broadcasting semantics
 */

class ngraph::pass::FakeQuantizeReshapeFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FakeQuantizeReshapeFusion();
};
