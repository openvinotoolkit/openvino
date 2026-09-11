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

/// This transformation handles the non-constant case where a binary element-wise op
/// consumes the output of a unit-dimension Unsqueeze/Reshape whose producer is a
/// "fusable" op (one that can absorb the eltwise as a post-op, e.g. FullyConnected,
/// MatMul, Transpose). Instead of reshaping the other input (which is not a constant),
/// it squeezes the other input's unit dimension, performs the eltwise in the lower rank,
/// and unsqueezes the result:
///     Eltwise(R, Unsqueeze(P, axis))
///       => Unsqueeze(Eltwise(Squeeze(R, axis), P), axis)
///
/// so the eltwise operates in the lower rank and can be fused as a post-op on P.
class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMovFusableProducer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveEltwiseUpThroughDataMovFusableProducer");
    /// @param fusable_producer_types List of op types that can fuse an eltwise
    ///        as a post-op. Passed as a runtime vector of type infos (instead of
    ///        compile-time wrap_type<...>()) because callers may include plugin-private
    ///        ops that are not visible to this device-agnostic common transformation,
    ///        e.g. ov::intel_gpu::op::FullyConnected. The common library must not depend
    ///        on a plugin, so those types can only be injected at runtime.
    /// @param check_bias_add If true, also considers Add(fusable_op, bias) as a fusable
    ///        producer (for cases where a bias add is fused into the preceding op's kernel).
    MoveEltwiseUpThroughDataMovFusableProducer(std::vector<DiscreteTypeInfo> fusable_producer_types,
                                               bool check_bias_add = false);
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
