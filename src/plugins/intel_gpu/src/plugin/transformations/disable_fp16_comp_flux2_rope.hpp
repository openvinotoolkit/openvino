// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Before f32->f16 ConvertPrecision, tag FLUX.2-style decomposed RoPE cos/sin table
 *        subgraphs (RoPEFusionFlux pattern) so frequency / trig tables stay in FP32.
 *
 * Matches the unfused y = x*cos + rotate_half(x)*sin graph and walks backward from the cos
 * and sin tensor roots into each multiply, mirroring MarkRopeInputsToKeepInMixedPrecision.
 */
class DisableFP16CompFlux2RoPEPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompFlux2RoPEPattern");
    DisableFP16CompFlux2RoPEPattern();

private:
    std::unordered_set<ov::Node*> m_visited;
};

}  // namespace ov::intel_gpu
