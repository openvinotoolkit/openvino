// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DivideFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief DivideFusion transformation replaces a sub-graph
 * Pow(y, -1) * x or x * Pow(y, -1) with Divide(x,y)
 */
class ov::pass::DivideFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DivideFusion", "0");
    DivideFusion();
};
