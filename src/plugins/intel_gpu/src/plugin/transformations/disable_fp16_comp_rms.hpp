// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief This transformation disables fp16 compression for RMS nodes in a specific pattern
 * to prevent precision loss.
 *
 * The targeted pattern is:
 *
 *     ...               ...
 *      |                 |
 *   Add (f32)        RMS (f32)
 * (add_m)          (rms_post_m)
 *      \              /
 *       \            /
 *         Add (f32)
 *        (add_1_m)
 *            |
 *            |
 *         RMS (f32)
 *         (rms_m)
 *
 * This pass finds the final RMS node (rms_m) in this chain and disables fp16 compression
 * for both itself and the preceding RMS node (rms_post_m). This is done to maintain
 * higher precision, as the result of the intermediate `add_1_m` operation can exceed
 * the representable range of fp16, leading to significant precision loss.
 * By keeping this pattern in fp32, numerical stability is preserved.
 */
class DisableFP16CompForGemma3RMSPattern: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForGemma3RMSPattern");
    DisableFP16CompForGemma3RMSPattern();
};

}   // namespace ov::intel_gpu
