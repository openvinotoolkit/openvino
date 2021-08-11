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

class TRANSFORMATIONS_API LeakyReluFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief LeakyReluFusion transformation replaces following graph:
 * Multiply->Maximum to LeakyRelu
 */

class ov::pass::LeakyReluFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LeakyReluFusion();
};
