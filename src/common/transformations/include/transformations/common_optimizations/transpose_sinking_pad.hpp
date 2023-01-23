// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingPadForward;
class TRANSFORMATIONS_API TransposeSinkingPadBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingPadForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingPadForward", "0");
    TransposeSinkingPadForward();
};

class ov::pass::TransposeSinkingPadBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingPadBackward", "0");
    TransposeSinkingPadBackward();
};
