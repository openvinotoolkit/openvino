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
class ngraph::pass::MulFakeQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulFakeQuantizeFusion", "0");
    MulFakeQuantizeFusion();
};
