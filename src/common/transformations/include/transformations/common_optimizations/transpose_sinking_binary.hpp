// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingBinaryForward;
class TRANSFORMATIONS_API TransposeSinkingBinaryBackward;
class TRANSFORMATIONS_API TransposeSinkingConcatForward;
class TRANSFORMATIONS_API TransposeSinkingConcatBackward;
class TRANSFORMATIONS_API TransposeSinkingSplitForward;
class TRANSFORMATIONS_API TransposeSinkingSplitBackward;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::TransposeSinkingBinaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ngraph::pass::TransposeSinkingBinaryForward", "0");
    TransposeSinkingBinaryForward();
};

class ngraph::pass::TransposeSinkingBinaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ngraph::pass::TransposeSinkingBinaryBackward", "0");
    TransposeSinkingBinaryBackward();
};

class ngraph::pass::TransposeSinkingConcatForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ngraph::pass::TransposeSinkingConcatForward", "0");
    TransposeSinkingConcatForward();
};

class ngraph::pass::TransposeSinkingConcatBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ngraph::pass::TransposeSinkingConcatBackward", "0");
    TransposeSinkingConcatBackward();
};

class ngraph::pass::TransposeSinkingSplitForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ngraph::pass::TransposeSinkingSplitForward", "0");
    TransposeSinkingSplitForward();
};

class ngraph::pass::TransposeSinkingSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ngraph::pass::TransposeSinkingSplitBackward", "0");
    TransposeSinkingSplitBackward();
};
