// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * Folds constant weight splits before decompression to keep their outputs compressible:
 * Constant -> decompression Convert -> Split/VariadicSplit becomes one low-precision
 * Constant -> decompression Convert per output, without materializing FP32 weights.
 */
class MoveDecompressionAfterSplit : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MoveDecompressionAfterSplit");
    MoveDecompressionAfterSplit();
};

}  // namespace ov::frontend::pytorch::pass
