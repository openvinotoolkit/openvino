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
 * - Uses TopK indices from router to dynamically gather expert weights
 * - No NonZero/ScatterElementsUpdate (device-friendly operations)
 * - Keeps full computation in-graph without splitting
 * - Reduces host-device communication compared to HOST_ROUTED mode
 *
 * Transformation Strategy:
 * 1. Locate Router's TopK node (selecting K active experts)
 * 2. Find grouped expert weights constants [num_experts, ...]
 * 3. Insert Gather nodes using TopK indices to select active expert weights
 * 4. Replace batched expert computations with dynamic weight selection
 * 5. Keep Multiply + ReduceSum for final aggregation with sparse scores
 */

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Transform batched MoE experts to use Gather-based dynamic weight selection
 *
 * Pattern to match:
 *   Router: Input → MatMul → Add → TopK (output 0: scores, output 1: indices)
 *           TopK → Softmax → ... routing processing
 *
 *   Experts: Tile → Reshape → MatMul(grouped_weights) → ... expert computation
 *            ... → Multiply(expert_outputs × routing_scores) → ReduceSum
 *
 * Transformation:
 *   Router: TopK indices → used for dynamic Gather
 *
 *   Experts: For each grouped weight constant [num_experts, d1, d2]:
 *            - Insert Gather(weights, TopK_indices, axis=0)
 *            - Output shape: [K, d1, d2] where K = top_k value
 *            - Replace batched computation with dynamic K-expert computation
 *
 * This preserves the Multiply × ReduceSum pattern but with dynamically selected weights.
 */
class DeviceRoutedMoETransform : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("npuw::pass::DeviceRoutedMoETransform", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/**
 * @brief High-level pass orchestrating device-routed MoE transformations
 *
 * Applies DeviceRoutedMoETransform across the model to convert all MoE layers
 * to use Gather-based dynamic expert selection.
 */
class DeviceRoutedMoEOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("npuw::pass::DeviceRoutedMoEOptimization", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
