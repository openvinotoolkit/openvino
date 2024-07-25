// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/// This transformation tries to put element-wise operations (Unary or Binary with scalar second input) before a set of data movement ops
/// in order to allow further element-wise op fusion to previous op and zero-copy optimizations
/// for data movement op itself.
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
    OPENVINO_RTTI("MoveEltwiseUpThroughDataMovScalar", "0");
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
    OPENVINO_RTTI("MoveEltwiseUpThroughDataMovPerChannel", "0");
    MoveEltwiseUpThroughDataMovPerChannel();
};

class TRANSFORMATIONS_API MoveEltwiseUpThroughDataMov : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MoveEltwiseUpThroughDataMov", "0");
    MoveEltwiseUpThroughDataMov(std::vector<DiscreteTypeInfo> allowed_data_movement_ops = get_default_allowed_ops()) {
        this->add_matcher<MoveEltwiseUpThroughDataMovScalar>(allowed_data_movement_ops);
        this->add_matcher<MoveEltwiseUpThroughDataMovPerChannel>();
    }

private:
    static std::vector<DiscreteTypeInfo> get_default_allowed_ops();
};

}  // namespace pass
}  // namespace ov
