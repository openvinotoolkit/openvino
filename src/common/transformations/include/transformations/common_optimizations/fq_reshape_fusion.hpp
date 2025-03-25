// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FakeQuantizeReshapeFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation looks for a FQ + Reshape pair in the graph and moves
 * the Reshape operation above the FQ node. Shapes of limit inputs are updated
 * following FQ broadcasting semantics
 */

class ov::pass::FakeQuantizeReshapeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FakeQuantizeReshapeFusion");
    FakeQuantizeReshapeFusion();
};
