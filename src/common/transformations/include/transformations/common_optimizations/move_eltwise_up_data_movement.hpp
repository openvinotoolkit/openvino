// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/// This transformation tries to put element-wise operations (Unary or Binary with scalar second input) before a set of
/// data movement ops in order to allow further element-wise op fusion to previous op and zero-copy optimizations for
/// data movement op itself.
///     ┌───────────┐                           ┌───────────┐
///     │   AnyOp   │                           │   AnyOp   │
///     └─────┬─────┘                           └─────┬─────┘
///           │                                       │
///           │                                       │
///   ┌───────┴────────┐                      ┌───────┴────────┐
///   | DataMovementOp |          =>          |  Element-Wise  |
///   └───────┬────────┘                      └───────┬────────┘
///           │                                       │
///           │                                       │
///   ┌───────┴────────┐                      ┌───────┴────────┐
///   │  Element-Wise  |                      │ DataMovementOp |
///   └────────────────┘                      └────────────────┘
class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMovScalar : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveEltwiseUpThroughDataMovScalar");
    MoveEltwiseUpThroughDataMovScalar(std::vector<DiscreteTypeInfo> allowed_data_movement_ops);
};

/// This transformation tries to put element-wise operations before Reshape/Squeeze/Unsqueeze ops
/// when second input to eltwise is per-channel Constant op
///     ┌───────────┐       ┌────────────────┐                    ┌───────────┐            ┌────────────────────┐
///     │   AnyOp   │       │  TargetShape   │                    │   AnyOp   │            │  Per-Channel Const │
///     └─────┬─────┘       └────────┬───────┘                    └─────┬─────┘            └─────────┬──────────┘
///           │                      │                                  │                            │
///           │                      |                                  │                            |
///           │   ┌─────────┐        │                                  │   ┌──────────────┐         │
///           └───┤ Reshape ├────────┘                    =>            └───┤ Element-Wise ├─────────┘
///               └────┬────┘                                               └───────┬──────┘
///                    │                                                            │
///                    │                                                            │
///            ┌───────┴────────┐    ┌────────────────────┐                   ┌─────┴─────┐    ┌─────────────┐
///            │  Element-Wise  ├────┤  Per-Channel Const │                   │  Reshape  ├────┤ TargetShape │
///            └────────────────┘    └────────────────────┘                   └───────────┘    └─────────────┘
class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMovPerChannel : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveEltwiseUpThroughDataMovPerChannel");
    MoveEltwiseUpThroughDataMovPerChannel();
};

/// This transformation moves a binary element-wise op below a data movement op (Unsqueeze or
/// rank-increasing Reshape) that inserts a single unit dimension, when the producer of that op can
/// fuse the eltwise as a post-op (e.g. FullyConnected, MatMul, Transpose). The unit dimension of the
/// other eltwise input is squeezed instead, so that the eltwise is computed in the lower rank:
///        ┌───────────┐  ┌──────────────┐              ┌───────────┐           ┌────────────┐
///        │ FusableOp │  │  TargetShape │              │ FusableOp │           │ OtherInput │
///        └─────┬─────┘  └──────┬───────┘              └─────┬─────┘           └─────┬──────┘
///              │               │                            │                       │
///      ┌───────┴───────────────┴───────┐                    │              ┌────────┴───────┐
///      │      Reshape / Unsqueeze      │       =>           │              │    Squeeze     │
///      └───────┬───────────────────────┘                    │              └────────┬───────┘
///              │                                      ┌─────┴───────────────────────┴─────┐
///      ┌───────┴────────┐   ┌────────────┐            │            Element-Wise           │
///      │  Element-Wise  ├───┤ OtherInput │            └────────────────┬──────────────────┘
///      └────────────────┘   └────────────┘                     ┌───────┴────────┐
///                                                              │   Unsqueeze    │
///                                                              └────────────────┘
/// The producer may be followed by a bias Add, which belongs to it because the bias is folded into the
/// producer's kernel. The data movement op has to be the first input of a non-commutative eltwise,
/// because the eltwise is fused into the producer as `producer <op> peer` post-op, which would reverse
/// the operands otherwise. A Reshape is accepted only when its shapes prove a pure dimension insertion,
/// which excludes more than one dynamic dimension.
class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMovFusableProducer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveEltwiseUpThroughDataMovFusableProducer");
    /// @param fusable_producer_types Types of the ops that can fuse an eltwise as a post-op. Passed as
    ///        runtime type infos because callers may pass plugin-private ops, e.g. GPU FullyConnected.
    MoveEltwiseUpThroughDataMovFusableProducer(std::vector<DiscreteTypeInfo> fusable_producer_types);
};

class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMov : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("MoveEltwiseUpThroughDataMov");
    MoveEltwiseUpThroughDataMov(std::vector<DiscreteTypeInfo> allowed_data_movement_ops = get_default_allowed_ops()) {
        this->add_matcher<MoveEltwiseUpThroughDataMovScalar>(allowed_data_movement_ops);
        this->add_matcher<MoveEltwiseUpThroughDataMovPerChannel>();
    }

private:
    static std::vector<DiscreteTypeInfo> get_default_allowed_ops();
};

}  // namespace pass
}  // namespace ov
