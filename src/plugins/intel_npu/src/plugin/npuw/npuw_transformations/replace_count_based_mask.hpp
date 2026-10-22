// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Rewrites a COUNT-based attention mask into a POSITION-based one.
//
// Models exported with ONNX Runtime's GroupQueryAttention reconstruct the attention mask
// from the number of attended tokens instead of from their positions.  The decomposed
// subgraph (one per layer) looks like:
//
//     total    = ReduceSum(attention_mask)      // a scalar - positions are lost here
//     past'    = total - S                      // S = query window length
//     bound[i] = past' + i + 1
//     mask     = (j < bound[i]) ? 0 : -inf      // j = key index
//
// That is only correct when  sum(attention_mask) == past_len + S,  i.e. when neither the
// query window nor the key window carries any slack.  CPU and GPU satisfy it for free
// (dynamic shapes, windows sized to the actual sequence).  NPUW runs static windows and
// violates it in BOTH stages:
//
//   prefill:  a variable-length prompt is left-padded into a fixed max_prompt_len query
//             window, so past' goes negative and every real query row is shifted onto the
//             padding region - real tokens attend to pad KV and never to each other.
//             Output is garbage.
//
//   generate: the past KV window is fixed at (total_size - S) while only num_stored_tokens
//             slots hold real keys.  The newly appended token lives at the END of the key
//             axis, past the reconstructed bound, so it cannot attend to itself and one
//             stale slot is attended instead.  A token that cannot see itself does not
//             know it was emitted, so the model re-emits it - visible as every token
//             appearing twice ("The The", "is is").
//
// The pass replaces the condition with a position-aware equivalent.  S (query window) and
// T (key window) are static by the time it runs (after ReshapeToStatic), and the runtime
// right-aligns the query window inside the key window, so query row i owns key position
// (T - S) + i and the causal term is a compile-time constant:
//
//     cond = (j <= (T-S) + i)  AND  attention_mask[j]      // position-aware
//            OR (j == (T-S) + i)                           // keep pad rows well-defined
//
// The second term matters only for query rows that carry no real token (prompt padding in
// prefill, unused generation slots in generate), whose output is discarded anyway: it
// leaves them exactly one visible key instead of an all -inf softmax row, avoiding a NaN
// that could otherwise leak into later layers.
//
// One rewrite covers every stage, because all of them right-align the query window:
//   * prefill          - S == T, left-padded prompt;
//   * chunked prefill  - S == chunk size, past chunks at key slots [0, num_stored);
//   * generate         - S == max_generation_token_len, past at [0, T-S), new tokens at
//                        the tail.  With S == 1 the causal term is all-true and the
//                        condition collapses to attention_mask[j] OR self, exactly the
//                        valid-past-plus-current-token set.
//
// The condition is built once per model and shared by all layers, so partitioning sees a
// single common subexpression (and the per-layer [1,H,S,T] i32 CumSum bound chain becomes
// dead).  The rewritten mask is [1,..,1,S,T] instead of [1,H,S,T] - it is broadcast into
// the scores by the consuming Add, which also drops an H-fold materialization of the mask.
//
// The pass is strictly self-guarding: it only fires on a Select(cond, 0, -inf) whose
// condition is Less(exclusive key-index CumSum, bound) with bound derived from
// ReduceSum(attention_mask).  Exports that build the mask from positions (optimum-intel:
// Gather + BitwiseAnd + SDPA) do not match and are left untouched.
//
// NOTE: the durable fix belongs in the exporter - it should emit a position-based mask
// (or an attention op with an explicit mask input) rather than rely on
// sum(attention_mask) == past_len + query_len, which no static-shape backend can satisfy.
class ReplaceCountBasedMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ReplaceCountBasedMask");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
