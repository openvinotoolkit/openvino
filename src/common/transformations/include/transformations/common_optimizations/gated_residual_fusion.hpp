// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
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
 * The callback is asked about the fused op; returning true keeps the matched sub-graph (a READ with a rejected write
 * gate is retried without it). Compressed weights (Constant -> Convert -> Subtract(zp) -> Multiply(scale)) are fed in
 * the op's [N, G] scale layout. The graph stays untouched unless the per-branch norm, the branch-axis reduction and
 * both 1/C scalings are proven.
 */
class TRANSFORMATIONS_API GatedResidualReadFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedResidualReadFusion");
    GatedResidualReadFusion();
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::pass::pattern::Matcher> m_matcher;
    ov::matcher_pass_callback m_fuse;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Replaces the Gated Residual WRITE pattern with ov::op::internal::GatedResidual in Mode::WRITE.
 *
 * Canonical sub-graph:
 *   R'   = Reshape(Reshape(R, [T,C,D]) + Reshape(S,[T,C,1]) * Reshape(Y,[T,1,D]), [T, C*D])
 *
 * The gate S is an opaque [*, C] input; the `2 * Sigmoid(...)` that produces it stays outside.
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
 * @brief Runs GatedResidualReadFusion, GatedResidualWriteSinkReshape and GatedResidualWriteFusion.
 */
class TRANSFORMATIONS_API GatedResidualFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedResidualFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
