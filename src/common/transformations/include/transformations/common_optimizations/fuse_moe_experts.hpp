// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/multi_matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @brief Data structure representing a single expert's weight parameters and metadata.
 *
 * Used during MoE fusion to track and organize expert-specific information across
 * the pattern matching and transformation process.
 */
struct expert_data {
    std::shared_ptr<Node> gate_proj_weight;  ///< Gate projection weight matrix
    std::shared_ptr<Node> up_proj_weight;    ///< Up projection weight matrix
    std::shared_ptr<Node> down_proj_weight;  ///< Down projection weight matrix
    size_t expert_id;                        ///< Expert identifier (0-based index)
    std::shared_ptr<Node> permute_node;      ///< Associated permute/transpose node for MoE layer grouping
};

/**
 * @brief Detects decomposed Mixture-of-Experts (MoE) layers whose experts are expressed
 * as independent SWIGLU 3-GEMM pipelines with scatter-based accumulation, and replaces
 * them with a single fused representation.
 *
 * The pass traverses each matched MoE layer, extracts per-expert gate/up/down weights,
 * preserves optional fp16->fp32 decompression Converts, concatenates the parameters into
 * expert-major tensors, and rebuilds the batched 3-GEMM computation together with the
 * routing weights tensor. The resulting fused subgraph mirrors the structure expected by
 * vectorized MoE passes.
 */
class TRANSFORMATIONS_API FuseMOEExperts : public ov::pass::MultiMatcher {
public:
    OPENVINO_RTTI("FuseMOEExperts");
    FuseMOEExperts();
};

/**
 * @brief High-level MoE transformation that invokes @c FuseMOEExperts across the
 * entire model, replacing decomposed expert blocks with the fused 3-GEMM
 * representation compatible with vectorized MoE passes.
 */
class TRANSFORMATIONS_API FuseMOE : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseMOE");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
