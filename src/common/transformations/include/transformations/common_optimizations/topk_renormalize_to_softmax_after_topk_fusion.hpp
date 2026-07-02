// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API TopkRenormalizeToSoftmaxAfterTopkFusion;

}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief Replaces (Softmax -> TopK -> renormalize) with (TopK -> Softmax).
 *
 * Pattern:
 *
 *   x -> Softmax(axis=A) -> TopK(k, axis=A).values
 *                              |
 *                          ReduceSum(axis=A, keep_dims=true)
 *                              |
 *                          Divide(values, sum)
 *
 * Identity used:
 *   softmax(x).topk_values / sum(softmax(x).topk_values)
 *     == exp(x_topk) / sum(exp(x_topk)) == softmax(topk(x).values, axis=A)
 *
 * After:
 *
 *   x -> TopK(k, axis=A) -> values -> Softmax(axis=A)
 *                       \-> indices (unchanged)
 *
 * Net: drops the upstream Softmax, ReduceSum, and Divide; adds a Softmax
 * over k elements.
 *
 * Must run before ConvertDivide so the matcher can anchor on Divide.
 */
class ov::pass::TopkRenormalizeToSoftmaxAfterTopkFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TopkRenormalizeToSoftmaxAfterTopkFusion");
    TopkRenormalizeToSoftmaxAfterTopkFusion();
};
