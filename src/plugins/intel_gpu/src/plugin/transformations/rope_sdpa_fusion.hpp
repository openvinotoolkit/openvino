// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Deletes the interleaved RoPE that feeds SDPA's Q by handing the cos/sin table to SDPA as two
/// extra trailing inputs; micro-SDPA then rotates Q inside the tile load it already performs.
class RoPESDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPESDPAFusion");
    RoPESDPAFusion();
};

}  // namespace ov::intel_gpu
