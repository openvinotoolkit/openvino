// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * @interface CanonicalizeFloatPrecision
 * @brief Rewrites a bf16/f16 model into f32 activations with weights kept narrow behind a Convert(f32).
 *
 * Model inputs/outputs and "__pa__" Parameters keep their type; models with subgraph ops are left unchanged.
 * Run by normalize() when the decoder's default float type isn't f32; MarkCompressedFloatConstants, run after
 * it, marks the new Converts as decompression.
 */
class CanonicalizeFloatPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::CanonicalizeFloatPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::frontend::pytorch::pass
