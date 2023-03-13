// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingInterpolateForward;
class TRANSFORMATIONS_API TransposeSinkingInterpolateBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingInterpolateForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingInterpolateForward", "0");
    TransposeSinkingInterpolateForward();
};

class ov::pass::TransposeSinkingInterpolateBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingInterpolateBackward", "0");
    TransposeSinkingInterpolateBackward();
};
