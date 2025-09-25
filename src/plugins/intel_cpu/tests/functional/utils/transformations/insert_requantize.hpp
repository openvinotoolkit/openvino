// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/pass/matcher_pass.hpp"
#include "utils/quantization_utils.hpp"

namespace CPUTestUtils {

/**
 * This pass inserts FakeQuantize + ShuffleChannels before Result.
 * ShuffleChannels ensures that only quantization part will be fused into the parent node.
 * It is also just a data movement operation which preserves the shapes.
 * This allows testing of quantized output.
 *
 * Before:
 *       *
 *       |
 *     Result
 *
 * After:
 *       *
 *       |
 *      FQ
 *       |
 * ShuffleChannels
 *       |
 *     Result
 *
 */
class InsertRequantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("InsertRequantize");
    InsertRequantize(size_t input_id, const QuantizationData& qinfo);
};

}  // namespace CPUTestUtils
