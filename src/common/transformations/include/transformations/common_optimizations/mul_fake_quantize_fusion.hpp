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

class TRANSFORMATIONS_API MulFakeQuantizeFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MulFakeQuantizeFusion transformation replaces following graph:
 * Mul->FakeQuantize to a single FakeQuantize
 * Restrictions:
 * - second input to Mul is a Constant
 */
class ngraph::pass::MulFakeQuantizeFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MulFakeQuantizeFusion();
};
