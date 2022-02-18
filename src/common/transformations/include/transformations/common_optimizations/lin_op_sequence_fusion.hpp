// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API LinOpSequenceFusion;
class TRANSFORMATIONS_API AddMultiplyFusion;
class TRANSFORMATIONS_API AddAddFusion;
class TRANSFORMATIONS_API MultiplyMultiplyFusion;

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

/**
 * @ingroup ie_transformation_common_api
 * @brief LinOpSequenceFusion transformation fuses linear operation sequence.
 */
class ngraph::pass::LinOpSequenceFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    LinOpSequenceFusion() {
        add_matcher<ngraph::pass::AddMultiplyFusion>();
        add_matcher<ngraph::pass::AddAddFusion>();
        add_matcher<ngraph::pass::MultiplyMultiplyFusion>();
    }
};