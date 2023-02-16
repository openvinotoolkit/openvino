// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingBinaryForward;
class TRANSFORMATIONS_API TransposeSinkingBinaryBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingBinaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBinaryForward", "0");
    TransposeSinkingBinaryForward();
};

class ov::pass::TransposeSinkingBinaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBinaryBackward", "0");
    TransposeSinkingBinaryBackward();
};
