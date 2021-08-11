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

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinking;
class TRANSFORMATIONS_API TransposeReduction;
class TRANSFORMATIONS_API TransposeFQReduction;
class TRANSFORMATIONS_API TransposeFuse;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeReduction transformation sinks Transpose through Reduce operations
 */
class ov::pass::TransposeReduction : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeReduction();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeFQReduction transformation sinks Transpose through FakeQuantize in case it is followed by reduction or squeeze
 */
class ov::pass::TransposeFQReduction : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeFQReduction();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeFuse transformation eliminates 2 consequtive Transposes if they result in no changes to input or fuses them
 * to single Transpose if input gets changed
 */
class ov::pass::TransposeFuse : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeFuse();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinking transformation sinks Transposes through known operations
 */
class ov::pass::TransposeSinking: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeSinking() {
        add_matcher<ov::pass::TransposeFQReduction>();
        add_matcher<ov::pass::TransposeReduction>();
        add_matcher<ov::pass::TransposeFuse>();
    }
};
