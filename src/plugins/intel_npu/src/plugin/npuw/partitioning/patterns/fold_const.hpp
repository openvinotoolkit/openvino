// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

// Lightweight constant-folding passes for shape-compute chains that appear in
// extracted MoE expert subgraphs after RegularizeSDPA folds ShapeOf(param)
// into a Constant.
//
// Target chain (each step enables the next):
//   ShapeOf(param)  --[RegularizeSDPA/ShapeOfParameter]--> Const
//   ShapeOf(any)    --[FoldShapeOf]--> Const   (bound-based, no input constraint)
//   Gather(C,C,C)   --[FoldGatherOfConst]--> Const
//   Unsqueeze(C,C)  --[FoldUnsqueezeOfConst]--> Const
//   Concat(C...)    --[FoldConcatOfConsts]--> Const
//
// FoldEltwiseOfConsts (Sub/Add/Mul/Div of two Constants) is a separate opt-in
// matcher, NOT part of FoldShapeComputeChain, so existing chain callers are
// unaffected; add it explicitly where an arithmetic step must also collapse.

namespace ov {
namespace npuw {
namespace patterns {
namespace util {

// Fold ShapeOf(any) → Constant when the output tensor has a known upper bound.
// More general than RegularizeSDPA::ShapeOfParameter: no constraint on input type.
class FoldShapeOf : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldShapeOf");
    FoldShapeOf();
};

// Fold Gather(Constant, Constant, Constant) → Constant.
class FoldGatherOfConst : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldGatherOfConst");
    FoldGatherOfConst();
};

// Fold Unsqueeze(Constant, Constant) → Constant.
class FoldUnsqueezeOfConst : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldUnsqueezeOfConst");
    FoldUnsqueezeOfConst();
};

// Fold Concat whose every input is a Constant → Constant.
// The pattern matches any Concat; the callback enforces the all-constant
// precondition so no variant for each arity is needed.
class FoldConcatOfConsts : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldConcatOfConsts");
    FoldConcatOfConsts();
};

// Fold a binary element-wise op (Add/Subtract/Multiply/Divide) whose both
// inputs are scalar integer Constants → Constant. Shape-compute chains frequently
// derive one split size as (total - other) etc.; without folding the arithmetic
// the downstream Concat never becomes all-constant. Not included in
// FoldShapeComputeChain by default (add it explicitly where needed); the
// scalar-integer-operand guard means it can never fold a weight-sized constant.
class FoldEltwiseOfConsts : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldEltwiseOfConsts");
    FoldEltwiseOfConsts();
};

// Runs the full shape-compute-chain folding pipeline in a single pass:
// FoldShapeOf → FoldGatherOfConst → FoldUnsqueezeOfConst → FoldConcatOfConsts.
class FoldShapeComputeChain : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("npuw::patterns::util::FoldShapeComputeChain");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Fold shape-compute chains (ShapeOf -> Gather -> Sub/Add -> Concat, ...) into
// Constants before online partitioning creates subgraph boundaries. Otherwise a
// partition boundary can cut such a chain and lift the (statically known) value
// into a runtime input Parameter; ops that require a Constant attribute (e.g.
// VariadicSplit's split_lengths from aten::split_with_sizes) then fail to compile
// on backends like VPUX. Folding only the ShapeOf root is NOT enough: the
// boundary can still cut the chain further down (e.g. at the Gather), leaving
// that op's input as a runtime Parameter. The whole chain is therefore collapsed
// to a single Constant here. FoldShapeOf is bound-based, so the fold is a no-op
// on genuinely dynamic shapes where no bound is resolvable.
//
// Unlike FoldShapeComputeChain, this also runs the opt-in FoldEltwiseOfConsts so a
// Subtract/Add step (split size = total - other) collapses too, and it only runs
// when a VariadicSplit with a non-constant split_lengths is actually present -
// staying a strict no-op (zero graph changes) for every model without the problem.
void foldShapeComputeChainsForConstAttrs(const std::shared_ptr<ov::Model>& model);

}  // namespace util
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
