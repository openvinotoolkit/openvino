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

class TRANSFORMATIONS_API FakeQuantizeMulFusion;

} // namespace pass
} // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief This transformation looks for a FQ + Mul pair in the graph and moves
 * the Mul operation above the FQ node. The last two inputs of FQ are multiplied
 * by the value that was originally below the FQ node.
 */

class ngraph::pass::FakeQuantizeMulFusion : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  FakeQuantizeMulFusion();
};
