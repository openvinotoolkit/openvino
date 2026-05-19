// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file device_routed_moe_transform.hpp
 * @brief Device-routed MoE transformation using Gather-based expert selection
 *
 * This transformation implements DEVICE_ROUTED mode for MoE models, where expert
 * selection is performed dynamically on the device using Gather operations driven
 * by Router's TopK outputs, avoiding graph splitting and reducing host-device overhead.
 *
 * Key Features:
 * - Uses TopK indices from router to dynamically gather expert weights and biases
 * - No NonZero/ScatterElementsUpdate (device-friendly operations)
 * - Keeps full computation in-graph without splitting
 * - Reduces host-device communication compared to HOST_ROUTED mode
 *
 * Transformation Strategy (Two-Phase Approach):
 * Phase 1 - Collection:
 *   1. Locate Router's TopK node (selecting K active experts per token)
 *   2. Extract TopK indices and Softmax scores
 *   3. Collect all expert nodes for the layer:
 *      - Tile nodes (expert dimension expansion)
 *      - Reshape nodes (constant or dynamic/unsqueeze-like)
 *      - MatMul nodes (expert computation with grouped weights)
 *      - Add nodes (expert biases)
 *      - Transpose nodes (routing score processing)
 *
 * Phase 2 - Transformation (all-or-nothing):
 *   1. Update Tile repeat counts from num_experts to K
 *   2. Update Reshape shapes to use K instead of num_experts
 *   3. Replace dynamic reshapes with Unsqueeze operations
 *   4. Insert Gather on expert weights/scales (for MatMul inputs)
 *   5. Insert Gather on expert biases (for Add inputs)
 *   6. Replace routing scores with TopK Softmax outputs
 *
 * Quantization Support:
 *   - Detects Multiply nodes in weight path (quantized_weight * scale)
 *   - Inserts Gather on both quantized weights and per-expert scales
 *   - Preserves Convert nodes for data type handling
 */

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Transform batched MoE experts to use Gather-based dynamic expert selection
 *
 * Pattern to match:
 *   Router:
 *     Input → MatMul(router_weights) → Add(router_bias) → TopK(K=num_active_experts)
 *     TopK.output(0): top-K scores  → Softmax → routing weights
 *     TopK.output(1): top-K indices → used for Gather operations
 *
 *   Experts (batched execution for all num_experts):
 *     Tile(repeat=[num_experts, 1, ...]) → Reshape([num_experts, ...])
 *     → MatMul(grouped_weights[num_experts, d1, d2]) → Add(grouped_bias[num_experts, d])
 *     → ... expert computation ...
 *     → Multiply(expert_outputs × routing_scores) → ReduceSum
 *
 * Transformation:
 *   Router:
 *     TopK.output(1) → Reshape([K]) → used as Gather indices
 *     TopK.output(0) → Softmax → replaces ScatterElementsUpdate routing scores
 *
 *   Experts (dynamic execution for K active experts):
 *     - Tile(repeat=[K, 1, ...])  // reduced from num_experts to K
 *     - Reshape([K, ...])          // updated shape
 *     - Gather(grouped_weights, topk_indices, axis=0) → [K, d1, d2]
 *     - Gather(grouped_bias, topk_indices, axis=0) → [K, d]
 *     - For quantized weights: Gather both weight and scale tensors
 *     - Dynamic reshapes replaced with Unsqueeze operations
 *
 * This transformation reduces computation from num_experts (e.g., 32) to K (e.g., 4)
 * active experts per token, with expert selection performed on-device via Gather.
 */
class DeviceRoutedMoETransform : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("npuw::pass::DeviceRoutedMoETransform", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
