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
 *                                                             |   |
 *                                                          Convert Convert (-> FP16)
 *
 * The Converts end the FP32 region at the tables, so their consumers (e.g. a RoPE rotation)
 * stay in FP16. They are not added in front of Results, Converts, or consumers that another
 * pass has already kept in FP32. Consumers marked later, inside ConvertPrecision, read the tables
 * rounded to FP16.
 */
class DisableFP16CompForDirectMultiplySinCos : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForDirectMultiplySinCos");
    DisableFP16CompForDirectMultiplySinCos();
};

}  // namespace ov::intel_gpu
