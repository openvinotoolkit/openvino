// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API TransposeSinking;
class TRANSFORMATIONS_API TransposeConvert;
class TRANSFORMATIONS_API TransposeEltwise;
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
    OPENVINO_RTTI("TransposeReduction", "0");
    TransposeReduction();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeFQReduction transformation sinks Transpose through FakeQuantize in case it is followed by reduction
 * or squeeze
 */
class ngraph::pass::TransposeFQReduction : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeFQReduction", "0");
    TransposeFQReduction();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeConvert transformation sinks Transpose through Convert
 */
class ngraph::pass::TransposeConvert : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeConvert", "0");
    TransposeConvert();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeEltwise transformation sinks Transpose through Eltwise
 */
class ngraph::pass::TransposeEltwise : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeEltwise", "0");
    TransposeEltwise();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeFuse transformation eliminates 2 consequtive Transposes if they result in no changes to input or
 * fuses them to single Transpose if input gets changed
 */
class ngraph::pass::TransposeFuse : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeFuse", "0");
    TransposeFuse();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinking transformation sinks Transposes through known operations
 */
class ngraph::pass::TransposeSinking : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeSinking", "0");
    TransposeSinking() {
        add_matcher<ngraph::pass::TransposeFQReduction>();
        add_matcher<ngraph::pass::TransposeReduction>();
        add_matcher<ngraph::pass::TransposeConvert>();
        add_matcher<ngraph::pass::TransposeEltwise>();
        add_matcher<ngraph::pass::TransposeFuse>();
    }
};
