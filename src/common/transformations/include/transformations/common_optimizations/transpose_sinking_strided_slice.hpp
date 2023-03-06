// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingStridedSliceForward;
class TRANSFORMATIONS_API TransposeSinkingStridedSliceBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingStridedSliceForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingStridedSliceForward", "0");
    TransposeSinkingStridedSliceForward();
};

class ov::pass::TransposeSinkingStridedSliceBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingStridedSliceBackward", "0");
    TransposeSinkingStridedSliceBackward();
};
