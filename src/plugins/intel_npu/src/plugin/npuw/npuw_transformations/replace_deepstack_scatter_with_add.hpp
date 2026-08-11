// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Replaces the Qwen3-VL "DeepStack" per-position scatter injection with a plain
// residual add and drops the now-unused visual_pos_masks model input.
//
// In the source model each deepstack level L is injected into the hidden states only
// at the visual-token positions via
//     ScatterNDUpdate(hidden, pos, GatherND(hidden, pos) + Gather(deepstack, L)).
// NPUW runs a static pipeline where the host can instead pre-scatter the deepstack
// values into their sequence positions, so the graph only needs
//     hidden + Gather(deepstack, L)
// (zeros at non-visual positions make the add a no-op there). This removes the
// data-dependent NonZero/GatherND/ScatterNDUpdate ops and the visual_pos_masks input,
// while keeping deepstack_visual_embeds as a single [num_layers, seq, emb] input.
//
// This is a ModelPass (not a MatcherPass) because it removes a model parameter; the
// node-level rewrite is done by an inner MatcherPass.
class ReplaceDeepstackScatterWithAdd : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ReplaceDeepstackScatterWithAdd");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
