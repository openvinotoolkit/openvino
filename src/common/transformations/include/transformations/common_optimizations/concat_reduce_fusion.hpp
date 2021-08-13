// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API PullSqueezeThroughEltwise;
class TRANSFORMATIONS_API ConcatReduceFusionWithoutFolding;
class TRANSFORMATIONS_API ConcatReduceFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConcatReduceFusion transformation replaces following graph:
 *
 *               +---------------+            +---------------+
 *               │               │            |               |
 *               │     input     │            |     input     |
 *               │               │            |               |
 *               +---------------+            +----------------
 *                       |                            |
 *                       |                            |
 *                       \                            /
 *                        \                          /
 *                         \                        /
 *                          \                      /
 *                           \                    /
 *                            \                  /
 *                             \                /
 *                              \              /
 *                               \            /
 *                              +---------------+
 *                              |               |
 *                              |     Concat    |
 *                              |               |
 *                              +----------------
 *                                      |
 *                                      v
 *                              +---------------+
 *                              |               |
 *                              |   ReduceMin/  |
 *                              |   ReduceMax   |
 *                              +----------------
 *
 * to a single Minimum/Maximum with 2 inputs.
 */


class ngraph::pass::PullSqueezeThroughEltwise: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullSqueezeThroughEltwise();
};

class ngraph::pass::ConcatReduceFusionWithoutFolding: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConcatReduceFusionWithoutFolding();
};


class ngraph::pass::ConcatReduceFusion: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
