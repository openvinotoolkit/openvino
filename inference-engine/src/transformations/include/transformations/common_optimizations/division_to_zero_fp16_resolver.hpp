// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DivisionToZeroFP16Resolver;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief :

 */
class ngraph::pass::DivisionToZeroFP16Resolver: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DivisionToZeroFP16Resolver();
};
