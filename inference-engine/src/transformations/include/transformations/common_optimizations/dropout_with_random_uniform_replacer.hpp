// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DropoutWithRandomUniformReplacer;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief This transformation replaces possible Dropout block (in inference mode) with RandomUniform
    to Broadcast of half-ones in a sub-graph.
 *
 */
class ngraph::pass::DropoutWithRandomUniformReplacer : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DropoutWithRandomUniformReplacer();
};
