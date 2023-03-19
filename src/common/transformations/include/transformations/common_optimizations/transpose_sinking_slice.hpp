// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingSliceForward;
class TRANSFORMATIONS_API TransposeSinkingSliceBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingSliceForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSliceForward", "0");
    TransposeSinkingSliceForward();
};

class ov::pass::TransposeSinkingSliceBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSliceBackward", "0");
    TransposeSinkingSliceBackward();
};
