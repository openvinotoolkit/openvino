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
 * pass has already kept in FP32. Consumers that ConvertPrecision keeps in FP32 later are
 * reconnected to the FP32 tables by RestoreFP32SinCosForFP32Consumers.
 */
class DisableFP16CompForDirectMultiplySinCos : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForDirectMultiplySinCos");
    DisableFP16CompForDirectMultiplySinCos();
};

/**
 * @brief Runs after ConvertPrecision. A consumer of an FP32 Sin/Cos that ConvertPrecision kept in FP32
 *        reads it through the FP16 Convert that ends the FP32 region and the Convert back to FP32 that
 *        AlignMixedFP32FP16Types puts in front of a kept node. This connects the consumer to the FP32
 *        Sin/Cos output directly.
 *
 *  Sin/Cos (FP32) -> Convert (FP16) -> Convert (FP32, kept) -> consumer   =>   Sin/Cos (FP32) -> consumer
 */
class RestoreFP32SinCosForFP32Consumers : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RestoreFP32SinCosForFP32Consumers");
    RestoreFP32SinCosForFP32Consumers();
};

}  // namespace ov::intel_gpu
