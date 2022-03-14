// Copyright (C) 2018-2022 Intel Corporation
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
 *  to Broadcast of half-ones in a sub-graph.
 *
 *   Dropout block:
 *   RandomUniform ----------> Add --->  Floor
 *   /\        /\              /\
 *   |         |               |
 *  Const(0)  Const(1)        Const(1)
 *  min_val   max_val
 *
 *  Resulted block:
 *  Broadcast -------> Add ---> Floor
 *    /\               /\
 *    |                |
 *  Const(0.5)      Const(1)
 *
 */
class ngraph::pass::DropoutWithRandomUniformReplacer : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DropoutWithRandomUniformReplacer", "0");
    DropoutWithRandomUniformReplacer();
};
