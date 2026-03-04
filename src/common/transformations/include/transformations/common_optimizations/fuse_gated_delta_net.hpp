// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GatedDeltaNetFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses gated delta net sub-graph into internal LinearAttention operation.
 */
class ov::pass::GatedDeltaNetFusion : public ov::pass::MatcherPass {
public:
	OPENVINO_MATCHER_PASS_RTTI("GatedDeltaNetFusion");
	GatedDeltaNetFusion();
};