// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API TransposeSinking;
class TRANSFORMATIONS_API TransposeReduction;
class TRANSFORMATIONS_API TransposeFQReduction;
class TRANSFORMATIONS_API TransposeFuse;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeReduction transformation sinks Transpose through Reduce operations
 */
class ngraph::pass::TransposeReduction : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeReduction();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeFQReduction transformation sinks Transpose through FakeQuantize in case it is followed by reduction or squeeze
 */
class ngraph::pass::TransposeFQReduction : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeFQReduction();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeFuse transformation eliminates 2 consequtive Transposes if they result in no changes to input or fuses them
 * to single Transpose if input gets changed
 */
class ngraph::pass::TransposeFuse : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeFuse();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinking transformation sinks Transposes through known operations
 */
class ngraph::pass::TransposeSinking: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeSinking() {
        add_matcher<ngraph::pass::TransposeFQReduction>();
        add_matcher<ngraph::pass::TransposeReduction>();
        add_matcher<ngraph::pass::TransposeFuse>();
    }
};
