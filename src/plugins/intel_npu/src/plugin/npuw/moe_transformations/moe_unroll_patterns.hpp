// Copyright (C) 2018-2026 Intel Corporation
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
// UnrollMoEMatMul: Unified MoE MatMul unrolling for all input patterns
// =============================================================================
/**
 * @brief Unrolls MoE expert MatMul operations into separate expert branches
 *
 * Automatically detects and handles three input patterns:
 * Pattern 1 (Batched):  input_param → convert → tile → reshape ─┐
 *                       scale + weights → multiply ─────────────┤ MatMul
 * Pattern 2 (Concat):   Concat([a,b,c,d]) ──────────────────────┐
 *                       scale + weights → multiply ─────────────┤ MatMul
 * Pattern 3 (Sliceable): AnyInput[N,...] ───────────────────────┐ (auto-sliced)
 *                       scale + weights → multiply ─────────────┤ MatMul
 *
 * Number of experts is automatically detected from scale/weights parameter shapes.
 * Transforms to: N expert branches with individual parameters, concatenated output
 *
 * @param model Model to register new parameters with
 */

class UnrollMoEMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::UnrollMoEMatMul");
    explicit UnrollMoEMatMul(std::shared_ptr<ov::Model> model);

private:
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
// UnrollParameterMultiply: Unroll Multiply with parameter when other input is not Concat
// =============================================================================
/**
 * @brief Unrolls Multiply operation when one input is a batched Parameter and the other is not Concat (e.g., AWQ
 * Multiply where Swish is the other input)
 *
 * Handles cases like: Multiply(k_parameter[N,...], Swish) where Swish is not unrolled
 * The parameter must be unrolled for correct weight loading in partial unroll scenarios.
 *
 * Transforms: Multiply(Param[N,...], NonConcat) → Concat([Multiply(Param[0], Slice(NonConcat,0)),
 *                                                          Multiply(Param[1], Slice(NonConcat,1)),
 *                                                          ...,
 *                                                          Multiply(Param[N-1], Slice(NonConcat,N-1))])
 *
 * Requirements:
 * - One input must be a Parameter with first dimension N > 1 (possibly through Convert)
 * - Other input must NOT be a Concat (to avoid conflicts with PushMultiplyBeforeConcat)
 * - Other input must have compatible shape for slicing on axis 0
 *
 * @param model Model to register split parameters with
 */

class UnrollParameterMultiply : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::UnrollParameterMultiply");
    explicit UnrollParameterMultiply(std::shared_ptr<ov::Model> model);

private:
    std::shared_ptr<ov::Model> model_;
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
// RemoveUnusedParameters: Clean up unused parameters after unrolling
// =============================================================================
/**
 * @brief Removes parameters that have no consumers after MoE unrolling
 *
 * After unrolling batched parameters into per-expert parameters,
 * the original batched parameters may become unused. This pass removes them.
 *
 * @param model Model to clean up
 */

class RemoveUnusedParameters : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::RemoveUnusedParameters");
    explicit RemoveUnusedParameters(std::shared_ptr<ov::Model> model);

private:
    std::shared_ptr<ov::Model> model_;
};

// =============================================================================
// Main transformation: Combine all MoE unrolling patterns
// =============================================================================
/**
 * @brief Full MoE expert unrolling with comprehensive optimizations
 *
 * Applies all available unrolling and optimization patterns including both
 * weight-related and activation-related transformations:
 *
 * Weight-related patterns:
 * - UnrollMoEMatMul: Unified MatMul unrolling for all input patterns
 * - PushElementwiseBeforeConcat: Distribute Add/Multiply operations
 * - PushMultiplyBeforeConcat: Fuse two Concat with Multiply
 * - PushReshapeBeforeConcat: Distribute Reshape on axis 0
 * - FuseConcatReduceSum: Convert Concat+ReduceSum to Add chain
 *
 * Activation-related patterns:
 * - PushSliceBeforeConcat: Distribute Slice when safe
 * - PushClampBeforeConcat: Distribute Clamp operations
 * - PushScalarElementwiseBeforeConcat: Distribute scalar operations
 *
 * Usage:
 * @code
 * auto model = std::make_shared<ov::Model>(...);
 * ov::pass::Manager manager;
 * manager.register_pass<ov::npuw::pass::MoEExpertUnrolling>(model);
 * manager.run_passes(model);
 * @endcode
 *
 * @param model Model instance for parameter registration
 */

class MoEExpertUnrolling : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("npuw::pass::MoEExpertUnrolling");

    explicit MoEExpertUnrolling(std::shared_ptr<ov::Model> model) {
        // Weight-related unrolling patterns
        add_matcher<UnrollMoEMatMul>(model);
        add_matcher<PushElementwiseBeforeConcat>(model);
        add_matcher<PushMultiplyBeforeConcat>();
        add_matcher<PushReshapeBeforeConcat>();
        add_matcher<FuseConcatReduceSum>();

        // Activation-related optimization patterns
        add_matcher<PushSliceBeforeConcat>();
        add_matcher<PushClampBeforeConcat>();
        add_matcher<PushScalarElementwiseBeforeConcat>();

        // Cleanup
        add_matcher<RemoveUnusedParameters>(model);
    }
};

/**
 * @brief Weight-focused MoE expert unrolling for parameter optimization
 *
 * Applies only weight-related unrolling patterns, focusing on splitting
 * batched parameters and weight operations into per-expert branches.
 * This variant excludes activation-related optimizations (Slice, Clamp, etc.)
 * to minimize graph modifications while still achieving parameter unrolling.
 *
 * Included patterns:
 * - UnrollMoEMatMul: Unified MatMul unrolling for all input patterns
 * - PushElementwiseBeforeConcat: Distribute Add/Multiply with parameters
 * - PushMultiplyBeforeConcat: Fuse two Concat with Multiply
 * - PushReshapeBeforeConcat: Distribute Reshape on axis 0
 * - FuseConcatReduceSum: Convert Concat+ReduceSum to Add chain
 *
 * Use cases:
 * - When only parameter splitting is needed
 * - When activation graph should remain unchanged
 * - For more conservative optimization strategy
 *
 * @param model Model instance for parameter registration
 */

class MoEExpertUnrollingWeightsOnly : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("npuw::pass::MoEExpertUnrollingWeightsOnly");

    explicit MoEExpertUnrollingWeightsOnly(std::shared_ptr<ov::Model> model) {
        // Only weight-related patterns - no activation optimizations
        add_matcher<UnrollMoEMatMul>(model);
        add_matcher<PushElementwiseBeforeConcat>(model);
        add_matcher<PushMultiplyBeforeConcat>();
        add_matcher<PushReshapeBeforeConcat>();
        add_matcher<FuseConcatReduceSum>();
        add_matcher<RemoveUnusedParameters>(model);
    }
};

/**
 * @brief AWQ parameter multiply unrolling with cleanup
 *
 * Unrolls Multiply operations with batched parameters (e.g., AWQ quantization scales)
 * and removes unused parameters after unrolling. This is specifically designed for
 * handling AWQ multiply nodes where one input is a batched parameter and the other
 * is not a Concat (e.g., Swish activation).
 *
 * Included patterns:
 * - UnrollParameterMultiply: Unroll Multiply with batched parameter
 * - RemoveUnusedParameters: Clean up unused batched parameters
 *
 * Use cases:
 * - Used in combination with MoEExpertUnrollingWeightsOnly for AWQ quantized models
 * - Not needed with MoEExpertUnrolling (already handles all patterns comprehensively)
 *
 * @param model Model instance for parameter registration and cleanup
 */

class UnrollAWQParameterMultiply : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("npuw::pass::UnrollAWQParameterMultiply");

    explicit UnrollAWQParameterMultiply(std::shared_ptr<ov::Model> model) {
        add_matcher<UnrollParameterMultiply>(model);
        add_matcher<RemoveUnusedParameters>(model);
    }
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
