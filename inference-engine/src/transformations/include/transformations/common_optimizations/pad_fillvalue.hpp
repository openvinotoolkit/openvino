// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API PadFill;
class TRANSFORMATIONS_API PadFillValue;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief 
 * 
 */
class ngraph::pass::PadFillValue: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFillValue();
};

class ngraph::pass::PadFill: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    PadFill() {
        add_matcher<ngraph::pass::PadFillValue>();
    }
};
