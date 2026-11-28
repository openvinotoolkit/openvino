// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \note HyperConnectionPre op class is under development and subject to change
///
/// \brief The whole pre side of one manifold-constrained hyper-connection (mHC) site in one op: RMS norm of the
/// flattened streams, the hc_fn projection to the mixing logits, HyperConnectionMix, and the pre-weighted stream sum.
/// Replaces RMS + a tiny [n * e] -> [(2 + n) * n] MatMul + HyperConnectionMix + Multiply/ReduceSum per site, whose
/// launch latency dominates at decode.
///
/// For streams h[..., n, e] (n streams of width e), with x = flatten(h) over the last two axes:
///   x     = x / sqrt(mean(x^2) + rms_eps)
///   mixes = x * fn^T                                  fn [(2 + n) * n, n * e]
///   pre, post, comb = HyperConnectionMix(mixes, scale, base; n, sinkhorn_iterations, eps)
///   y     = sum_i pre[..., i] * h[..., i, :]
/// Outputs: y [..., e], post [..., n], comb [..., n, n] laid out [src][dst] (as HyperConnectionMix).
class OPENVINO_API HyperConnectionPre : public ov::op::Op {
public:
    OPENVINO_OP("HyperConnectionPre");

    HyperConnectionPre() = default;
    /// \param streams [..., n, e] residual streams.
    /// \param fn [(2 + n) * n, n * e] hc_fn projection weight.
    /// \param scale [3] scales for the pre, post and comb parts.
    /// \param base [(2 + n) * n] biases.
    HyperConnectionPre(const Output<Node>& streams,
                       const Output<Node>& fn,
                       const Output<Node>& scale,
                       const Output<Node>& base,
                       int64_t n_streams,
                       int64_t sinkhorn_iterations,
                       float eps,
                       float rms_eps);

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
    float get_rms_eps() const {
        return m_rms_eps;
    }

private:
    int64_t m_n_streams = 4;
    int64_t m_sinkhorn_iterations = 20;
    float m_eps = 1e-6F;
    float m_rms_eps = 1e-6F;
};

}  // namespace ov::op::internal
