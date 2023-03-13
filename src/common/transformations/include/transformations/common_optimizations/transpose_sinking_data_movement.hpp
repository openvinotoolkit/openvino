// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingDataMovementForward;
class TRANSFORMATIONS_API TransposeSinkingDataMovementBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingDataMovementForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingDataMovementForward", "0");
    TransposeSinkingDataMovementForward();
};

class ov::pass::TransposeSinkingDataMovementBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingDataMovementBackward", "0");
    TransposeSinkingDataMovementBackward();
};
