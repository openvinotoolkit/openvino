// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file moe_unroll_patterns.hpp
 * @brief Graph transformation patterns for Mixture of Experts (MoE) model optimization
 *
 * This file contains a collection of transformation patterns designed to optimize
 * MoE model graphs for execution on Intel NPU. The patterns focus on:
 * - Unrolling batched expert computations into parallel branches
 * - Distributing operations before concatenation for better parallelism
 * - Fusing operation sequences to reduce computational overhead
 *
 * Usage: Apply MoEExpertUnrolling pass to the model with specified number of experts
 */

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

// =============================================================================
// UnrollBatchedMatMul: Unroll batched MatMul to per-expert branches
// =============================================================================
/**
 * @brief Unrolls batched MoE expert MatMul operations into separate expert branches
 *
 * Matches pattern: input → convert → tile → reshape ─┐
 *                  scale + weights → multiply ─────┤ MatMul
 * Transforms to: N expert branches with individual parameters, concatenated output
 *
 * @param num_experts Number of expert branches to create
 * @param model Model to register new parameters with
 */

class UnrollBatchedMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::UnrollBatchedMatMul");
    explicit UnrollBatchedMatMul(size_t num_experts, std::shared_ptr<ov::Model> model);

private:
    size_t num_experts_;
    std::shared_ptr<ov::Model> model_;
};

// =============================================================================
// PushElementwiseBeforeConcat: Push elementwise operations before Concat
// =============================================================================
/**
 * @brief Distributes elementwise operations (Add/Multiply) before Concat
 *
 * Transforms: Param[N,...] + Concat([a,b,c]) → Concat([a+Param[0], b+Param[1], c+Param[2]])
 * Benefits: Enables per-branch optimization and reduces parameter size
 *
 * @param model Model to register split parameters with
 */

class PushElementwiseBeforeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::PushElementwiseBeforeConcat");
    explicit PushElementwiseBeforeConcat(std::shared_ptr<ov::Model> model);

private:
    std::shared_ptr<ov::Model> model_;
};

// =============================================================================
// PushSliceBeforeConcat: Push Slice before Concat
// =============================================================================
/**
 * @brief Distributes Slice operations before Concat when safe
 *
 * Safety: Only applies when slice axis != concat axis
 * Transforms: Concat([a,b,c], axis=0) → Slice(axis=k) → Concat([Slice(a), Slice(b), Slice(c)])
 */

class PushSliceBeforeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::PushSliceBeforeConcat");
    PushSliceBeforeConcat();
};

// =============================================================================
// PushClampBeforeConcat: Push Clamp before Concat
// =============================================================================
/**
 * @brief Distributes Clamp operations before Concat
 *
 * Transforms: Concat([a,b,c]) → Clamp(min,max) → Concat([Clamp(a), Clamp(b), Clamp(c)])
 * Always safe: Clamp operates element-wise independently
 */

class PushClampBeforeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::PushClampBeforeConcat");
    PushClampBeforeConcat();
};

// =============================================================================
// PushScalarElementwiseBeforeConcat: Push scalar elementwise operations before Concat
// =============================================================================
/**
 * @brief Distributes scalar operations (Add/Minimum/Swish) before Concat
 *
 * Constraint: Scalar input must have exactly 1 element
 * Transforms: Concat([a,b,c]) → Op(scalar) → Concat([Op(a,scalar), Op(b,scalar), Op(c,scalar)])
 */

class PushScalarElementwiseBeforeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::PushScalarElementwiseBeforeConcat");
    PushScalarElementwiseBeforeConcat();
};

// =============================================================================
// PushMultiplyBeforeConcat: Push Multiply before Concat
// =============================================================================
/**
 * @brief Fuses two Concat operations with element-wise Multiply
 *
 * Requirements: Same axis, same branch count, compatible shapes
 * Transforms: Concat([a,b,c]) * Concat([d,e,f]) → Concat([a*d, b*e, c*f])
 */

class PushMultiplyBeforeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::PushMultiplyBeforeConcat");
    PushMultiplyBeforeConcat();
};

// =============================================================================
// FuseConcatReduceSum: Fuse Concat + ReduceSum
// =============================================================================
/**
 * @brief Converts Concat followed by ReduceSum to cascaded Add operations
 *
 * Requirements: ReduceSum axis must match Concat axis
 * Handles both keep_dims=true and keep_dims=false
 * Transforms: Concat([a,b,c], axis=0) → ReduceSum(axis=0) → a + b + c
 */

class FuseConcatReduceSum : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::FuseConcatReduceSum");
    FuseConcatReduceSum();
};

// =============================================================================
// UnrollExpertReshape: Unroll expert Reshape operations
// =============================================================================
/**
 * @brief Unrolls batched Reshape after Tile into per-expert branches
 *
 * Requirement: Reshape output shape must be [num_experts, 1, hidden_dim]
 * Transforms: Tile → Reshape[N,1,H] → N×Reshape[1,1,H] → Concat
 *
 * @param num_experts Number of experts to unroll
 */

class UnrollExpertReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::UnrollExpertReshape");
    explicit UnrollExpertReshape(size_t num_experts);

private:
    size_t num_experts_;
};

// =============================================================================
// UnrollConcatMatMul: Unroll MatMul with Concat input and parameter weights
// =============================================================================
/**
 * @brief Unrolls MatMul with Concat input into per-expert branches
 *
 * Pattern: Concat([a,b,c,d]) ──────┐
 *          scale[N,A,1] + weights[N,A,B] → multiply ─┤ MatMul
 * Output: N branches with parameters [1,A,1] and [1,A,B]
 *
 * @param model Model to register new split parameters with
 */

class UnrollConcatMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::UnrollConcatMatMul");
    explicit UnrollConcatMatMul(std::shared_ptr<ov::Model> model);

private:
    std::shared_ptr<ov::Model> model_;
};

// =============================================================================
// PushReshapeBeforeConcat: Push Reshape before Concat
// =============================================================================
/**
 * @brief Distributes Reshape operations before Concat on axis 0
 *
 * Safety: Concat axis must be 0 and Reshape must preserve first dimension
 * Example: Concat([a,b,c,d], axis=0) → Reshape → Concat([Reshape(a), Reshape(b), ...])
 */

class PushReshapeBeforeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::PushReshapeBeforeConcat");
    PushReshapeBeforeConcat();
};

// =============================================================================
// Main transformation: Combine all MoE unrolling patterns
// =============================================================================
/**
 * @brief Comprehensive graph rewrite pass for MoE model optimization
 *
 * Combines all individual patterns into a single optimization pipeline:
 * - UnrollBatchedMatMul: Unroll batched expert MatMul operations
 * - UnrollConcatMatMul: Unroll MatMul with Concat inputs
 * - PushElementwiseBeforeConcat: Distribute Add/Multiply operations
 * - PushSliceBeforeConcat: Distribute Slice when safe
 * - PushClampBeforeConcat: Distribute Clamp operations
 * - PushScalarElementwiseBeforeConcat: Distribute scalar operations
 * - PushMultiplyBeforeConcat: Fuse two Concat with Multiply
 * - PushReshapeBeforeConcat: Distribute Reshape on axis 0
 * - FuseConcatReduceSum: Convert Concat+ReduceSum to Add chain
 *
 * Usage:
 * @code
 * auto model = std::make_shared<ov::Model>(...);
 * ov::pass::Manager manager;
 * manager.register_pass<ov::npuw::pass::MoEExpertUnrolling>(num_experts, model);
 * manager.run_passes(model);
 * @endcode
 *
 * @param num_experts Number of expert branches in the MoE model
 * @param model Model instance for parameter registration
 */

class MoEExpertUnrolling : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("npuw::pass::MoEExpertUnrolling");

    explicit MoEExpertUnrolling(size_t num_experts, std::shared_ptr<ov::Model> model) {
        add_matcher<UnrollBatchedMatMul>(num_experts, model);
        add_matcher<UnrollConcatMatMul>(model);
        add_matcher<PushElementwiseBeforeConcat>(model);
        add_matcher<PushSliceBeforeConcat>();
        add_matcher<PushClampBeforeConcat>();
        add_matcher<PushScalarElementwiseBeforeConcat>();
        add_matcher<PushMultiplyBeforeConcat>();
        add_matcher<PushReshapeBeforeConcat>();
        add_matcher<FuseConcatReduceSum>();
    }
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
