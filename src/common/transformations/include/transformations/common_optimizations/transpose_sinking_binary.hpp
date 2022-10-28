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
class TRANSFORMATIONS_API TransposeSinkingConcatForward;
class TRANSFORMATIONS_API TransposeSinkingConcatBackward;
class TRANSFORMATIONS_API TransposeSinkingSplitForward;
class TRANSFORMATIONS_API TransposeSinkingSplitBackward;
class TRANSFORMATIONS_API TransposeSinkingElementwiseForward;
class TRANSFORMATIONS_API TransposeSinkingElementwiseBackward;

}  // namespace pass
}  // namespace ngraph

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

class ov::pass::TransposeSinkingConcatForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingConcatForward", "0");
    TransposeSinkingConcatForward();
};

class ov::pass::TransposeSinkingConcatBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingConcatBackward", "0");
    TransposeSinkingConcatBackward();
};

class ov::pass::TransposeSinkingSplitForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSplitForward", "0");
    TransposeSinkingSplitForward();
};

class ov::pass::TransposeSinkingSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSplitBackward", "0");
    TransposeSinkingSplitBackward();
};

class ov::pass::TransposeSinkingElementwiseForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingElementwiseForward", "0");
    TransposeSinkingElementwiseForward();
};

class ov::pass::TransposeSinkingElementwiseBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingElementwiseBackward", "0");
    TransposeSinkingElementwiseBackward();
};
