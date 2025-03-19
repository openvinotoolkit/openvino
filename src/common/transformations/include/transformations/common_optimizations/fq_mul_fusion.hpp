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

class TRANSFORMATIONS_API FakeQuantizeMulFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation looks for a FQ + Mul pair in the graph and moves
 * the Mul operation above the FQ node. The last two inputs of FQ are multiplied
 * by the value that was originally below the FQ node.
 */

class ov::pass::FakeQuantizeMulFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FakeQuantizeMulFusion");
    FakeQuantizeMulFusion();
};
