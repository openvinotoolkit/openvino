// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MulFakeQuantizeFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief MulFakeQuantizeFusion transformation replaces following graph:
 * Mul->FakeQuantize to a single FakeQuantize
 * Restrictions:
 * - second input to Mul is a Constant
 */
class ov::pass::MulFakeQuantizeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulFakeQuantizeFusion", "0");
    MulFakeQuantizeFusion();
};
