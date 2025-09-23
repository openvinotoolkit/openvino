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

/**
 * @ingroup ov_transformation_common_api
 *
 * @brief KeepDequantizationPrecision matches Dequantization subgraphs and, if precision matches with
 * specified, Convert, Multiply, Subtract and Reshape nodes might be marked with disable_fp16_compression attribute.
 * This prevents precision loss when the original precision is lowered during ConvertPrecision execution.
 *
 * Example scenario:
 *      Original Dequantization subgraph:       Potential transformed subgraph after ConvertPrecision:
 *        Input (i32)      Const (i32)                    Input (i32)     Const (i32)
 *            │              │                               │              │
 *            ▼              ▼                               ▼              ▼
 *        Convert (f32)  Convert (f32)                   Convert (f16)  Convert (f16)
 *            │              │                               │              │
 *            ▼              ▼                               ▼              ▼
 *             Subtract (f32)                                 Subtract (f16)
 *                 │                                              │
 *                 │     Scale (f32)                              │     Scale (f16)
 *                 │      │                                       │      │
 *                 ▼      ▼                                       ▼      ▼
 *                 Multiply (f32)                                 Multiply (f16)
 *
 *    Without KeepDequantizationPrecision, ConvertPrecision transformation may convert
 *    these operations to use fp16 instead of f32, potentially leading to accuracy degradation.
 *    Marking these nodes (KeepDequantizationPrecision) preserves the original dequantization precision (f32).
 *
 */
class TRANSFORMATIONS_API KeepDequantizationPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KeepDequantizationPrecision");
    explicit KeepDequantizationPrecision(const element::TypeVector& precisions,
                                         bool add_precision_sensitive_convert = false);
};

/**
 * Marks subgraphs where Gather receives:
 *   - data: Constant in table_values_precisions → (optional Convert)
 *   - indices: Constant/Parameter in indices_precisions → (optional Convert)
 * Disables constant folding for Converts, enables keep-precision for Constants.
 *
 * Pattern:
 *
 *        [Constant]    [Constant/Parameter]   [Axis]
 *            |                  |                |
 *      (optional Convert) (optional Convert)     |
 *            |                  |                |
 *            +--------+---------+--------+-------+
 *                     |         |        |
 *              Gather (data, indices, axis)
 */
class TRANSFORMATIONS_API MarkGatherSubgraph : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkGatherSubgraph")
    MarkGatherSubgraph(const element::TypeVector& table_values_precisions,
                       const element::TypeVector& indices_precisions);
};

class TRANSFORMATIONS_API markStatefulSubgraph : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("markStatefulSubgraph")
    markStatefulSubgraph();
                       
};


}  // namespace pass
}  // namespace ov
