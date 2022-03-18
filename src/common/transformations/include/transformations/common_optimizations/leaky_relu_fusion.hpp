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

class TRANSFORMATIONS_API LeakyReluFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief LeakyReluFusion transformation replaces following graph:
 * Multiply->Maximum to LeakyRelu
 */

class ngraph::pass::LeakyReluFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("LeakyReluFusion", "0");
    LeakyReluFusion();
};
