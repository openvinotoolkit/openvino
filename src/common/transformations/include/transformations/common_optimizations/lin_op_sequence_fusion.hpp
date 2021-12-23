// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <openvino/core/visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API LinOpSequenceFusion;
class OPENVINO_API AddMultiplyFusion;
class OPENVINO_API AddAddFusion;
class OPENVINO_API MultiplyMultiplyFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::AddMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddMultiplyFusion();
};

class ngraph::pass::AddAddFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddAddFusion();
};

class ngraph::pass::MultiplyMultiplyFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyMultiplyFusion();
};

class ngraph::pass::LinOpSequenceFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    LinOpSequenceFusion() {
        add_matcher<ngraph::pass::AddMultiplyFusion>();
        add_matcher<ngraph::pass::AddAddFusion>();
        add_matcher<ngraph::pass::MultiplyMultiplyFusion>();
    }
};
