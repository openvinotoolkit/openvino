// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Recovers rotary embedding inverse frequency constant precision from f16 to f32.
 *
 * When models are exported in FP16, the inv_freq constant used to compute rotary position embeddings
 * gets compressed to f16. This causes significant phase errors in cos/sin computation for large
 * position_ids values (e.g., >100), leading to accuracy loss in models like Qwen2.5-VL, Qwen3-VL,
 * and others that compute position embeddings at runtime.
 *
 * The pass detects f16 inv_freq constants by finding the pattern:
 *   f16 Constant -> Convert(f16->f32) -> Broadcast -> MatMul -> ... -> Sin/Cos
 * It verifies the constant forms a geometric series (1/base^(2i/dim)),
 * recomputes it in f64 -> f32, and marks the computation chain with disable_fp16_compression.
 */
class TRANSFORMATIONS_API RecoverRoPEInvFreqPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RecoverRoPEInvFreqPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
