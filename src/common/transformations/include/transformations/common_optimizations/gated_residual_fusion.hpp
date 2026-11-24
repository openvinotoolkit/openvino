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
 * @brief Fuses the Gated Residual READ / MIX pattern, with its write gate if any, into
 * ov::op::internal::GatedResidual (Mode::READ).
 *
 *   rbar  = Reshape(RMS(Reshape(R, [T, C, D]), gamma, eps), [T, C*D])    // or the decomposed RMS form
 *   act   = Swish(MatMul(rbar, W_down, transpose_b) / C)
 *   mixed = ReduceSum(Reshape(Sigmoid(MatMul(act, W_up, transpose_b)) * rbar, [T, C, D]), axis=1) / C
 *   S     = 2 * Sigmoid(MatMul(rbar, W_inject, transpose_b) / C)         // optional write gate
 *   ==>  mixed [, S] = GatedResidual(R, gain, W_down, W_up [, W_inject])
 *
 * Left untouched unless the norm, the branch reduction and both 1/C scalings are proven. The pass callback may veto
 * the fused op.
 */
class TRANSFORMATIONS_API GatedResidualReadFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedResidualReadFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses the Gated Residual WRITE pattern into ov::op::internal::GatedResidual (Mode::WRITE).
 *
 *   R' = Reshape(Reshape(R, [T,C,D]) + Reshape(S,[T,C,1]) * Reshape(Y,[T,1,D]), [T, C*D])
 *
 * The gate S is an opaque [*, C] input. R, Y and S must have the same rows: equal static sizes or the same symbol
 * (propagate symbols first, e.g. with SymbolicOptimizations); unrelated dynamic dimensions are not fused.
 */
class TRANSFORMATIONS_API GatedResidualWriteFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedResidualWriteFusion");
    GatedResidualWriteFusion();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Sinks the Reshape of the Gated Residual WRITE pattern below the residual Add, so a backend can fuse the Add
 * and the Multiply as one eltwise.
 *
 *   Add(R, Reshape(S[.., C, 1] * Y[.., 1, D], [.., C*D]))
 *     -> Reshape(Add(Reshape(R, [.., C, D]), S[.., C, 1] * Y[.., 1, D]), [.., C*D])
 *
 * The pass callback may veto the match. Alternative to GatedResidualWriteFusion: a backend disables one of the two.
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
