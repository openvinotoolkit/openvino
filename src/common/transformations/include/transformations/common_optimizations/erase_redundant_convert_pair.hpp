// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// Eliminate redundant pairs of Convert nodes:
//   x (Tsrc) -> Convert(Tnarrow) -> Convert(Tsrc) -> consumer
// when Tsrc is wider than Tnarrow and the round-trip is value-preserving
// modulo the rounding to Tnarrow. We replace the outer Convert's output
// with the original source x, so the f32 (or wider) consumer sees the
// pre-narrowing value directly. The inner Convert is left in place for any
// other consumer that genuinely needs Tnarrow (typically the residual
// stream); DCE will remove it if it becomes unused.
//
// Generic across narrow float types (bf16, f16, etc.) and ISAs.
class TRANSFORMATIONS_API EraseRedundantConvertPair : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EraseRedundantConvertPair");
    EraseRedundantConvertPair();
};

}  // namespace ov::pass
