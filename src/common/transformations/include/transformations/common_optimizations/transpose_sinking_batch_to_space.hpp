// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingBatchToSpaceForward;
class TRANSFORMATIONS_API TransposeSinkingBatchToSpaceBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingBatchToSpaceForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBTSForward", "0");
    TransposeSinkingBatchToSpaceForward();
};

class ov::pass::TransposeSinkingBatchToSpaceBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBTSBackward", "0");
    TransposeSinkingBatchToSpaceBackward();
};
