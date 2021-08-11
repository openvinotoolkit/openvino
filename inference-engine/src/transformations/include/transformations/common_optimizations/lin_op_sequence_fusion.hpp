// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LinOpSequenceFusion;
class TRANSFORMATIONS_API AddMultiplyFusion;
class TRANSFORMATIONS_API AddAddFusion;
class TRANSFORMATIONS_API MultiplyMultiplyFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::AddMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddMultiplyFusion();
};

class ov::pass::AddAddFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddAddFusion();
};

class ov::pass::MultiplyMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyMultiplyFusion();
};

class ov::pass::LinOpSequenceFusion: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    LinOpSequenceFusion() {
        add_matcher<ov::pass::AddMultiplyFusion>();
        add_matcher<ov::pass::AddAddFusion>();
        add_matcher<ov::pass::MultiplyMultiplyFusion>();
    }
};
