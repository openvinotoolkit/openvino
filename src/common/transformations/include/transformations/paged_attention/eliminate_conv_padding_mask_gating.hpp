// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Eliminates the conv padding mask gating subgraph introduced by transformers 5.0+.
 *
 * Transformers 5.0 added `apply_mask_to_padding_states` which multiplies hidden_states
 * by attention_mask before conv layers. In PagedAttention mode, sequences are packed
 * contiguously without padding, so this gating is always an identity (mask is all-1s).
 *
 * Matches the pattern:
 *   attention_mask -> Slice -> Unsqueeze -> Convert -> Multiply -> Add -> Multiply(H, mask_expr)
 * and replaces the final Multiply with its hidden_states input directly.
 */
class TRANSFORMATIONS_API EliminateConvPaddingMaskGating : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateConvPaddingMaskGating");
    EliminateConvPaddingMaskGating();
};

}  // namespace pass
}  // namespace ov
