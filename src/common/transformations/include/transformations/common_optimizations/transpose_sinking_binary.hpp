// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingBinaryElementwiseForward;
class TRANSFORMATIONS_API TransposeSinkingBinaryElementwiseBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingBinaryElementwiseForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBinaryElementwiseForward", "0");
    TransposeSinkingBinaryElementwiseForward();
};

class ov::pass::TransposeSinkingBinaryElementwiseBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBinaryElementwiseBackward", "0");
    TransposeSinkingBinaryElementwiseBackward();
};
