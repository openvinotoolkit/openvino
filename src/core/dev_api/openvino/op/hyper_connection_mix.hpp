// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note HyperConnectionMix op class is under development and subject to change
///
/// \brief Turns the per-token mixing logits of a manifold-constrained hyper-connection (mHC, DeepSeek-V4 /
/// GLM-5.3) into its three coefficient sets, replacing a graph of several hundred tiny ops per call.
///
/// With n = hc streams, mixes[..., (2 + n) * n] is split into pre[n], post[n] and comb[n * n] (in that order):
///   pre  = sigmoid(mixes[0:n]   * scale[0] + base[0:n])   + eps
///   post = sigmoid(mixes[n:2n]  * scale[1] + base[n:2n])  * 2
///   c    = mixes[2n:] * scale[2] + base[2n:], viewed as c[src][dst]
///   c    = softmax over dst, + eps, then column-normalize (divide by sum over src + eps), then
///          (sinkhorn_iterations - 1) times: row-normalize (sum over dst + eps), column-normalize.
/// Outputs: pre [..., n], post [..., n], comb [..., n, n] laid out [src][dst].
/// Reference: the mHC formulation of DeepSeek-V4 / GLM-5.3-Flash (pre/post sigmoids, Sinkhorn-normalized comb).
class OPENVINO_API HyperConnectionMix : public ov::op::Op {
public:
    OPENVINO_OP("HyperConnectionMix");

    HyperConnectionMix() = default;
    /// \param mixes [..., (2 + n) * n] mixing logits (hc_fn applied to the RMS-normalized streams).
    /// \param scale [3] scales for the pre, post and comb parts.
    /// \param base [(2 + n) * n] biases.
    /// \param n_streams Number of residual streams n.
    /// \param sinkhorn_iterations Sinkhorn iterations (the first column normalization counts as one).
    /// \param eps Epsilon added to pre and inside the Sinkhorn normalizations.
    HyperConnectionMix(const Output<Node>& mixes,
                       const Output<Node>& scale,
                       const Output<Node>& base,
                       int64_t n_streams,
                       int64_t sinkhorn_iterations,
                       float eps);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

    int64_t get_n_streams() const {
        return m_n_streams;
    }
    int64_t get_sinkhorn_iterations() const {
        return m_sinkhorn_iterations;
    }
    float get_eps() const {
        return m_eps;
    }

private:
    int64_t m_n_streams = 4;
    int64_t m_sinkhorn_iterations = 20;
    float m_eps = 1e-6F;
};

}  // namespace ov::op::internal
