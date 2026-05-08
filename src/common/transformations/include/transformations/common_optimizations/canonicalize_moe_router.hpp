// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @brief Rewrites a MoE routing tail expressed via ScatterElementsUpdate
 * (densified routing weights) into the canonical Transpose + Gather form.
 *
 * Matches:
 *   ScatterElementsUpdate(zeros[T,E], topk_idx[T,K], topk_val[T,K], axis=1)
 *     -> Transpose([1,0]) -> Reshape -> (Unsqueeze(-1)) ->
 *     Multiply(Reshape(down_out, [E,B,*,H]), routing) ->
 *     ReduceSum(axis=0)
 *
 * Replaces with:
 *   Transpose(down_out, [1,0,2]) ->
 *   Gather(axis=1, batch_dims=1, topk_idx) ->
 *   Multiply(Unsqueeze(topk_val, -1)) ->
 *   ReduceSum(axis=1)
 *
 * Handles both fused-MoE output (FuseMOEExperts) and natively exported
 * GPT-OSS-style IR, which share the same ScatterElementsUpdate shape.
 */
class TRANSFORMATIONS_API CanonicalizeMoeRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CanonicalizeMoeRouter");
    CanonicalizeMoeRouter();
};

}  // namespace pass
}  // namespace ov
