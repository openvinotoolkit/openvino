// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class TransposeFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TransposeFusion");
    TransposeFusion(bool supports_immad = false);
};

class TransposeMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMulMatcher");
    TransposeMatMulMatcher(bool supports_immad);
};

class TransposeMatMulTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMulTransposeMatcher");
    TransposeMatMulTransposeMatcher(bool supports_immad);
};

class TransposeSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeSDPAMatcher");
    TransposeSDPAMatcher();
};

class TransposeVLSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeVLSDPAMatcher");
    TransposeVLSDPAMatcher();
};

class TransposeSplitMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeSplitMatcher");
    TransposeSplitMatcher();
};

/// Sinks a Transpose through the axis-removing split that unpacks a fused QKV tensor into
/// SDPA inputs, so that every SDPA input is fed by a lower-rank Transpose which
/// TransposeSDPAMatcher can absorb into the SDPA input order:
///
///   X -> Transpose(p) -> Split/VariadicSplit(axis a) -> Squeeze(a) -> [scalar Multiply] -> SDPA
///        (the Squeeze may also be a Reshape that only drops the unit axis a)
///   X -> Transpose(p) -> Gather(scalar index, axis a)              -> [scalar Multiply] -> SDPA
/// becomes
///   X -> Split/VariadicSplit(axis p[a]) -> Squeeze(p[a]) -> Transpose(p') -> ...
///   X -> Gather(scalar index, axis p[a])                 -> Transpose(p') -> ...
///
/// where p' is p without position a, renumbered to the reduced rank. Covers e.g.
/// qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0) and its qkv[i] form.
/// Must run before passes that insert ops between the split and SDPA (e.g. activations scaling).
/// A ModelPass because the rewrite replaces the consumers of the Transpose.
class SinkTransposeThroughSplitForSDPA : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SinkTransposeThroughSplitForSDPA");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}   // namespace ov::intel_gpu
