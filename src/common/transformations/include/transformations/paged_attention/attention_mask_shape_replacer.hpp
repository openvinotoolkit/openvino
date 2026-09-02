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
 * @brief Replaces ShapeOf(attention_mask) with ShapeOf(input_source) in the
 *        broadcast-shape branch that feeds rotary MatMul.
 *
 * Some models build a Broadcast target shape from attention_mask and then use that
 * Broadcast as the first input of MatMul, while the second input is derived from
 * position_ids through an optional Unsqueeze/Convert chain.
 * In PagedAttention mode, attention_mask is removed, therefore this shape branch
 * must be rewired to an always-present source (input_ids or inputs_embeds).
 *
 * Matched topology:
 *
 *  ┌─────────────────┐
 *  │ attention_mask  │
 *  │  (Parameter)    │
 *  └────────┬────────┘
 *           │
 *      ┌────┴────┐        ┌──────────────┐
 *      │ ShapeOf │        │ position_ids │
 *      └────┬────┘        │ (Parameter)  │
 *           │             └──────┬───────┘
 *      ┌────┴────┐         ┌─────┴──────┐
 *      │ Gather  │         │ Unsqueeze  │ (optional)
 *      └────┬────┘         └─────┬──────┘
 *           │               ┌────┴────┐
 *      ┌────┴────┐          │ Convert │ (optional)
 *      │ Concat  │          └────┬────┘
 *      └────┬────┘               │
 *           │                    │
 *      ┌────┴────┐               │
 *      │Broadcast├───────────────┘
 *      └────┬────┘
 *           │
 *      ┌────┴────┐
 *      │ MatMul  │
 *      └─────────┘
 *
 * Rewrite:
 *
 * Replace ShapeOf(attention_mask) with ShapeOf(input_source), where input_source
 * is input_ids or inputs_embeds.
 *
 * After:
 *
 *  ┌─────────────────┐
 *  │  input_source   │
 *  │  (Parameter)    │
 *  └────────┬────────┘
 *           │
 *      ┌────┴────┐        ┌──────────────┐
 *      │ ShapeOf │        │ position_ids │
 *      └────┬────┘        │ (Parameter)  │
 *           │             └──────┬───────┘
 *      ┌────┴────┐         ┌─────┴──────┐
 *      │ Gather  │         │ Unsqueeze  │ (optional)
 *      └────┬────┘         └─────┬──────┘
 *           │               ┌────┴────┐
 *      ┌────┴────┐          │ Convert │ (optional)
 *      │ Concat  │          └────┬────┘
 *      └────┬────┘               │
 *           │                    │
 *      ┌────┴────┐               │
 *      │Broadcast├───────────────┘
 *      └────┬────┘
 *           │
 *      ┌────┴────┐
 *      │ MatMul  │
 *      └─────────┘
 *
 * The rewrite is applied only when all of the following hold:
 *  1) Gather axis is 0 (or -1 for a 1D ShapeOf output).
 *  2) Gather indices are exactly {0}.
 *  3) input_source rank is static and at least 1.
 *
 * These checks guarantee that the selected dimension is equivalent after rewiring
 * and avoid altering unsupported shape semantics.
 */
class ov::pass::AttentionMaskShapeReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AttentionMaskShapeReplacer");
    explicit AttentionMaskShapeReplacer(const Output<Node>& input_source);
};
