// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AttentionMaskShapeReplacer;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Detaches the attention_mask parameter from shape-deriving subgraphs.
 *
 * Some models query the batch/sequence dimensions from the attention_mask shape.
 * In PagedAttention mode the attention_mask parameter is removed, so these shape
 * queries must be rewired to an always-present input (input_ids or inputs_embeds).
 * The attention_mask leading dimensions (batch, sequence) coincide with the leading
 * dimensions of input_ids / inputs_embeds, so the same Gather indices remain valid.
 *
 * Before:
 *
 *  ┌─────────────────┐
 *  │ attention_mask  │
 *  │  (Parameter)    │
 *  └────────┬────────┘
 *           │
 *      ┌────┴────┐   ┌──────────┐
 *      │ ShapeOf │   │ indices  │
 *      └────┬────┘   │ {0, 1 or │
 *           │        │  both}   │
 *           │        └────┬─────┘
 *        ┌──┴──┐──────────┘
 *        │Gather├──────────── axis
 *        └─────┘
 *
 * After (input_source is input_ids or inputs_embeds):
 *
 *  ┌─────────────────┐
 *  │  input_source   │
 *  │  (Parameter)    │
 *  └────────┬────────┘
 *           │
 *      ┌────┴────┐   ┌──────────┐
 *      │ ShapeOf │   │ indices  │
 *      └────┬────┘   │ {0, 1 or │
 *           │        │  both}   │
 *           │        └────┬─────┘
 *        ┌──┴──┐──────────┘
 *        │Gather├──────────── axis
 *        └─────┘
 *
 * The replacement is applied only when every requested Gather index is within the
 * rank of the provided input source; otherwise the match is skipped.
 */
class ov::pass::AttentionMaskShapeReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AttentionMaskShapeReplacer");
    explicit AttentionMaskShapeReplacer(const Output<Node>& input_source);
};
