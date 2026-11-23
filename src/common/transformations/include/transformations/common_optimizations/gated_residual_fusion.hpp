// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Replaces the Gated Residual READ / MIX pattern (optionally with its write-gate branch) with
 * ov::op::internal::GatedResidual in Mode::READ.
 *
 * Canonical sub-graph (token-major [T, C*D] activations):
 *   r3    = Reshape(R,   [T, C, D])
 *   n3    = RMS(r3, gamma, eps)                 // or the decomposed Power/ReduceMean/Add/Sqrt/Divide/Multiply form
 *   rbar  = Reshape(n3,  [T, C*D])
 *   low   = MatMul(rbar, W_down, transpose_b)   // W_down: [L, C*D]
 *   act   = Swish(low * 1/C)
 *   up    = MatMul(act,  W_up,   transpose_b)   // W_up:   [C*D, L]
 *   wm    = Sigmoid(up) * rbar
 *   mixed = ReduceSum(Reshape(wm, [T, C, D]), axis=1) * 1/C
 *
 * Replaced by (the write gate S = 2 * Sigmoid(MatMul(rbar, W_inject, transpose_b) / C) found on the same rbar is
 * fused as the second output):
 *   mixed [, S] = GatedResidual(R, gain, W_down, W_up [, W_inject])   // 1/C of both scalings applied by the op
 *
 * The normalization is matched in one of its forms (fused RMS, gamma-less RMS, decomposed), each by a matcher of its
 * own. The write gate is a branch outside the match, which the fused op replaces, so the pass is a ModelPass: the
 * branches are recorded first, then the READ matchers run. The callback is asked about the fused op; returning true
 * keeps the matched sub-graph (a READ with a rejected write gate is retried without it). The weights are fed to the op
 * as they are, a decompression subgraph of compressed ones included; a backend that runs them compressed converts the
 * op afterwards (ConvertGatedResidualToCompressed). The graph stays untouched unless the per-branch norm, the
 * branch-axis reduction and both 1/C scalings are proven.
 */
class TRANSFORMATIONS_API GatedResidualReadFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedResidualReadFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Replaces the Gated Residual WRITE pattern with ov::op::internal::GatedResidual in Mode::WRITE.
 *
 * Canonical sub-graph:
 *   R'   = Reshape(Reshape(R, [T,C,D]) + Reshape(S,[T,C,1]) * Reshape(Y,[T,1,D]), [T, C*D])
 *
 * The gate S is an opaque [*, C] input; the `2 * Sigmoid(...)` that produces it stays outside. The residual, Y and the
 * gate must have the same rows: equal static sizes or the same symbol, so the symbols have to be propagated through
 * the model first (e.g. by SymbolicOptimizations); unrelated dynamic dimensions may broadcast at runtime and are not
 * fused.
 */
class TRANSFORMATIONS_API GatedResidualWriteFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedResidualWriteFusion");
    GatedResidualWriteFusion();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Sinks the Reshape of the Gated Residual WRITE pattern below the residual Add, so the Add sits next to the
 * Multiply and a backend can fuse the two as one broadcasting eltwise.
 *
 *   Add(R, Reshape(S[.., C, 1] * Y[.., 1, D], [.., C*D]))
 *     -> Reshape(Add(Reshape(R, [.., C, D]), S[.., C, 1] * Y[.., 1, D]), [.., C*D])
 *
 * The gate S is an opaque [*, C] input; the `2 * Sigmoid(...)` that produces it stays outside. The callback is asked
 * about the matched Add; returning true keeps the original graph. The alternative to GatedResidualWriteFusion for
 * backends that run the chain as plain eltwise: disable one of the two.
 */
class TRANSFORMATIONS_API GatedResidualWriteSinkReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedResidualWriteSinkReshape");
    GatedResidualWriteSinkReshape();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Runs GatedResidualReadFusion, GatedResidualWriteSinkReshape and, optionally, GatedResidualWriteFusion.
 *
 * The WRITE pattern is either left to the backend as a plain eltwise chain with the Add next to the Multiply (the
 * sink), or fused into the op: a backend picks one with `fuse_write`.
 */
class TRANSFORMATIONS_API GatedResidualFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedResidualFusion");

    /// @param fuse_write Fuse the WRITE pattern into ov::op::internal::GatedResidual in Mode::WRITE; otherwise only
    /// move its Add (GatedResidualWriteSinkReshape).
    explicit GatedResidualFusion(bool fuse_write = true);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_fuse_write;
};

}  // namespace ov::pass
