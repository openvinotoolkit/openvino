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

class TRANSFORMATIONS_API ClampFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ClampFusion transformation replaces following graph:
 * Maximum->Minimum to Clamp
 * Restrictions:
 * - one of the parameters to Maximum is a scalar constant
 * - one of the parameters to Minimum is a scalar constant
 */

class ov::pass::ClampFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ClampFusion", "0");
    ClampFusion();
};
