// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 *
 * @brief MarkDequantization matches Dequantization subgraphs and marks Subtract and Multiply nodes
 * with the dequantization attribute. Also if Convert nodes are part of the subgraph they might be marked
 * with the disable_const_folding attribute.
 *
 * If Convert -> Reshape/Unsqueeze are part of the Dequantization subraph, Convert and Reshape/Unsqueeze
 * nodes will be swapped to eliminate Reshape/Unsqueeze in the next ConstantFolding.
 *
 * Dequantization subgraph may have two forms: with and without Subtract.
 * ZeroPoints and Scale might be present as subgraphs and include Convert ops.
 *
 *     Input       ZeroPoints
 *       │             │
 *       ▼             ▼
 *     Convert   (opt) Reshape/Unsqueeze
 *           │      │
 *           ▼      ▼    Scale                        Input     Scale
 *           Subtract     │                            │         │
 *                │       ▼                            ▼         ▼
 *                │     (opt) Reshape/Unsqueeze       Convert  (opt) Reshape/Unsqueeze
 *                │      │                               │      │
 *                ▼      ▼                               ▼      ▼
 *                Multiply                               Multiply
 *
 */
class TRANSFORMATIONS_API MarkDequantization : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkDequantization");
    explicit MarkDequantization(const element::TypeVector& precisions,
                                bool fold_subtract_const = false,
                                bool fold_multiply_const = true);
};

/**
 * @ingroup ov_transformation_common_api
 *
 * @brief KeepConstPrecision matches Dequantization subgraphs and if Input/ZeroPoints/Scale are Constants
 * they might be marked with keep_const_precision attribute.
 *
 * Dequantization subgraph may have two forms: with and without Subtract.
 *
 *        Input
 *          │
 *          ▼
 *       Convert  ZeroPoints
 *           │      │
 *           ▼      ▼                        Input
 *           Subtract                          │
 *                │                            ▼
 *                │     Scale               Convert   Scale
 *                │      │                     │      │
 *                ▼      ▼                     ▼      ▼
 *                Multiply                     Multiply
 *
 */
class TRANSFORMATIONS_API KeepConstPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KeepConstPrecision");
    explicit KeepConstPrecision(const element::TypeVector& precisions,
                                bool fold_subtract_const = false,
                                bool fold_multiply_const = true);
};

}  // namespace pass
}  // namespace ov
