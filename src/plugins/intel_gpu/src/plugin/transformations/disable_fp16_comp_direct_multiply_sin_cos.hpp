// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps a shared trigonometric phase calculation in FP32. Rounding the
 *        phase to FP16 before Sin/Cos causes unacceptable accuracy loss.
 *
 *  Before                              After
 *  ------                              -----
 *  input_0 (FP16) --\                  input_0 (FP32) --\
 *                    Multiply (FP16)                       Multiply (FP32)
 *  input_1 (FP16) --/       |          input_1 (FP32) --/       |
 *                         +-+-+                               +-+-+
 *                         |   |                               |   |
 *                        Sin Cos                             Sin Cos
 *                     (FP16) (FP16)                       (FP32) (FP32)
 *
 * Up to two shape-only ops (Reshape, Transpose, Squeeze, Unsqueeze) may sit between the Multiply and Sin/Cos, as in
 * split rotary embeddings (e.g. LTX-2: Multiply -> Transpose/Reshape -> Reshape -> Sin/Cos); they are kept in FP32 too.
 *
 * Only disable_fp16_compression rt_info is added; the graph topology is unchanged.
 */
class DisableFP16CompForDirectMultiplySinCos : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForDirectMultiplySinCos");
    DisableFP16CompForDirectMultiplySinCos();
};

}  // namespace ov::intel_gpu
