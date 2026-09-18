// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Keeps the decomposed LTX-Video RoPE angle chain (Multiply -> Add -> Transpose -> Reshape ->
 * Sin/Cos) in f32: the angles reach ~1.6e4 rad, which bf16 quantizes in steps larger than 2*pi.
 *
 * The CPU plugin registers this pass only under bf16 enforcement. The fused RoPE variant is handled
 * on GPU by IncreasePositionIdsPrecisionForLtxVideo.
 */
class TRANSFORMATIONS_API DisableBF16CompForLtxVideoRopePattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableBF16CompForLtxVideoRopePattern");
    DisableBF16CompForLtxVideoRopePattern();
};

}  // namespace pass
}  // namespace ov
