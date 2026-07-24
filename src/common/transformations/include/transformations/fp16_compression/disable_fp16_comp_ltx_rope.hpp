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
 * Sin/Cos) in f32: the angles reach ~1.6e4 rad, which f16/bf16 quantize in steps larger than 2*pi.
 *
 * The fused RoPE variant is handled on GPU by IncreasePositionIdsPrecisionForLtxVideo.
 */
class TRANSFORMATIONS_API DisableFP16CompForLtxVideoRopePattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForLtxVideoRopePattern");
    DisableFP16CompForLtxVideoRopePattern();
};

}  // namespace pass
}  // namespace ov
