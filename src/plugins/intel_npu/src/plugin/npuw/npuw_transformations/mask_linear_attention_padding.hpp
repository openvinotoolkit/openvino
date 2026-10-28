// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {

// Makes the recurrent state of linear-attention layers (GatedDeltaNet / Mamba-style /
// LFM2 short-conv mixers) insensitive to the left padding of a static-shape prefill chunk.
//
// NPUW feeds a partial prefill chunk left-padded: [pad ... pad | real tokens]. The model
// only zeroes the hidden states of the padded positions through the attention mask, which is
// not enough once a non-zero state is carried in from the previous chunk:
//   * conv:  Concat(past_conv, x) puts the padded positions between the carried history
//            and the first real token, so the first real tokens convolve over zeros instead
//            of the history, and the present conv state is built from the wrong columns.
//   * ssm:   a padded position still decays the state (g = -exp(A) * softplus(dt_bias) != 0)
//            and still updates it (beta = sigmoid(0) = 0.5, k and v can be non-zero where the
//            conv window overlaps the history).
//
// The pass requires static shapes (it runs after ReshapeToStatic) and a prefill batch of 1.
// All sequence lengths are read from the model, nothing is specific to a chunk size.
//
//   conv:  the Concat(past [B,C,K], x [B,C,L]) is followed by a ScatterUpdate that writes the history
//          over the K columns right in front of the first real token, pad + [0, K):
//          [h0 h1 h2 h3 0 0 x2 x3] -> [h0 h1 h0 h1 h2 h3 x2 x3] (K=4, L=4, pad=2).
//          Only K columns are written, no second tensor of the Concat's size is computed. The leading `pad`
//          columns are left as they are: they only feed the outputs of padded positions.
//          `pad = L - sum(attention_mask[:, -L:])` is computed in the graph, so the chunk
//          may be full (pad = 0, identity) or partial. Both the convolution and the present
//          state, which is the last K columns of the Concat, consume the re-ordered tensor.
//   ssm:   the per-token decay `g` and update gate `beta` feeding the recurrence Loop are
//          multiplied by the mask of the chunk, so a padded position is an identity step
//          (exp(0) * state, no update).
//
// The pass is all-or-nothing: if any conv or ssm state of the model is not recognized, the model
// is left unchanged and false is returned. A recurrence Loop is recognized by its body: g feeds
// [Convert] -> Exp -> Unsqueeze -> Multiply with the carried state, beta only scales the delta that is
// added to that decayed state, and the sum (optionally converted) is what the next iteration starts from
// (the GatedDeltaNet layout), so a zero g and beta is an identity step.
class MaskLinearAttentionPadding : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::MaskLinearAttentionPadding");

    // Returns true when every conv/ssm state of the model has been made padding-safe.
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace npuw
}  // namespace ov
