// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NGramHashBlockFusion;
class TRANSFORMATIONS_API NGramHashConcatFusion;
class TRANSFORMATIONS_API NGramHashFunctionFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses one n-gram order's hash-and-modulo block into an NGramHashFunction.
 *
 * Matches an int64 subgraph for one order, such as the Qwen3.8-Flash-Next PLE or DeepSeek Engram hash:
 *     mix   = (x[0] * m[0]) ^ (x[1] * m[1]) ^ ... ^ (x[o-1] * m[o-1])
 *     block = FloorMod(Unsqueeze(mix, -1), vocab) + offsets
 * and replaces it with Convert(NGramHashFunction(Concat(Unsqueeze(x[o-1]), ..., Unsqueeze(x[0])))). The generated
 * hash has heads only for this order; its leading heads_per_order entries are zero. The Convert preserves the
 * original i64 result type while keeping wrapping 64-bit multiplication and modulo inside the operation, before a
 * plugin can lower surrounding i64 arithmetic to i32.
 *
 * This pass handles each order independently, including blocks that are not part of a Concat. A later
 * NGramHashConcatFusion may merge compatible single-order results into one multi-order operation.
 */
class ov::pass::NGramHashBlockFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NGramHashBlockFusion");
    NGramHashBlockFusion();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Merges compatible single-order hash operations under an i64 Concat into one multi-order operation.
 *
 * Expects inputs produced by NGramHashBlockFusion: Convert(NGramHashFunction(...)) for strictly increasing n-gram
 * orders, concatenated on the last axis. Each order's token-and-multiplier terms must extend the previous order's
 * terms, so all heads can be computed from one packed token window. On success, replaces the Concat with
 * Convert(NGramHashFunction(...)) whose heads_per_order describes all concatenated orders.
 *
 * If the orders are out of order or do not share the required prefix terms, this pass leaves the Concat in place;
 * independent block fusions performed earlier remain valid.
 */
class ov::pass::NGramHashConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NGramHashConcatFusion");
    NGramHashConcatFusion();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses an int64 hashed n-gram graph into NGramHashFunction operation(s).
 *
 * Runs NGramHashBlockFusion first to replace each recognized order's hash-and-modulo block, then runs
 * NGramHashConcatFusion to combine compatible ascending orders that use the same extending token window. A graph
 * without a compatible multi-order Concat can still receive independent single-order fusions; unmatched or
 * incompatible subgraphs are left unchanged.
 */
class ov::pass::NGramHashFunctionFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("NGramHashFunctionFusion");
    NGramHashFunctionFusion() {
        add_matcher<NGramHashBlockFusion>();
        add_matcher<NGramHashConcatFusion>();
    }
};
