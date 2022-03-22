// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReplaceConcatReduceByMinOrMax;
class TRANSFORMATIONS_API PullSqueezeThroughEltwise;
class TRANSFORMATIONS_API ConcatReduceFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ReplaceConcatReduceByMinOrMax transformation replaces Concat with 2 inputs and ReduceMin/Max
 * by a single Minimum/Maximum with 2 inputs and inserts squeeze in case when Reduce has keep_dims = false.
 */
class ngraph::pass::ReplaceConcatReduceByMinOrMax : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReplaceConcatReduceByMinOrMax", "0");
    ReplaceConcatReduceByMinOrMax();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PullSqueezeThroughEltwise transformation propagates Squeeze up through binary elementwise operations:
 */
class ngraph::pass::PullSqueezeThroughEltwise : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("PullSqueezeThroughEltwise", "0");
    PullSqueezeThroughEltwise();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConcatReduceFusion pass replaces the following graph:
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
 * by a single Minimum/Maximum with 2 inputs and tries to eliminate Squeeze/Unsqueeze layers before and after Min/Max.
 */

class ngraph::pass::ConcatReduceFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConcatReduceFusion", "0");
    ConcatReduceFusion();
};
