// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReplaceConcatReduceByMinOrMax;
class TRANSFORMATIONS_API PullSqueezeThroughEltwise;
class TRANSFORMATIONS_API ConcatReduceFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReplaceConcatReduceByMinOrMax transformation replaces Concat with 2 inputs and ReduceMin/Max
 * by a single Minimum/Maximum with 2 inputs and inserts squeeze in case when Reduce has keep_dims = false.
 */
class ov::pass::ReplaceConcatReduceByMinOrMax : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReplaceConcatReduceByMinOrMax");
    ReplaceConcatReduceByMinOrMax();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief PullSqueezeThroughEltwise transformation propagates Squeeze up through binary elementwise operations:
 */
class ov::pass::PullSqueezeThroughEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PullSqueezeThroughEltwise");
    PullSqueezeThroughEltwise();
};

/**
 * @ingroup ov_transformation_common_api
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

class ov::pass::ConcatReduceFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConcatReduceFusion");
    ConcatReduceFusion();
};
