// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingSplitBackward;
class TRANSFORMATIONS_API TransposeSinkingBinaryElementwiseForward;
class TRANSFORMATIONS_API TransposeSinkingConcatForward;
class TRANSFORMATIONS_API TransposeSinkingSplitForward;
class TRANSFORMATIONS_API TransposeSinkingBinaryElementwiseBackward;
class TRANSFORMATIONS_API TransposeSinkingConcatBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingBinaryElementwiseForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBinaryElementwiseForward", "0");
    TransposeSinkingBinaryElementwiseForward();
};

class ov::pass::TransposeSinkingConcatForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingConcatForward", "0");
    TransposeSinkingConcatForward();
};

class ov::pass::TransposeSinkingSplitForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSplitForward", "0");
    TransposeSinkingSplitForward();
};

class ov::pass::TransposeSinkingBinaryElementwiseBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingBinaryElementwiseBackward", "0");
    TransposeSinkingBinaryElementwiseBackward();
};

class ov::pass::TransposeSinkingConcatBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingConcatBackward", "0");
    TransposeSinkingConcatBackward();
};

class ov::pass::TransposeSinkingSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSplitBackward", "0");
    TransposeSinkingSplitBackward();
};
