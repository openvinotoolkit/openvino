// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API Atan2Decomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Atan2Decomposition transformation replaces Atan2(y, x) with a quadrant-aware
 *        subgraph built from Atan, Divide, Select, and comparison ops, following
 *        the standard atan2 definition with correct IEEE 754 signed-zero handling.
 */
class ov::pass::Atan2Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Atan2Decomposition");
    Atan2Decomposition();
};
