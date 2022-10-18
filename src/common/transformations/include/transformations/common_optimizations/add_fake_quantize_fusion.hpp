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
class ngraph::pass::AddFakeQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddFakeQuantizeFusion", "0");
    AddFakeQuantizeFusion();
};
