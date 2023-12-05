// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

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

class ov::pass::LeakyReluFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LeakyReluFusion", "0");
    LeakyReluFusion();
};
