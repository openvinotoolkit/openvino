// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected.hpp"

namespace ov::intel_gpu::op {

class FullyConnectedCompressed : public FullyConnected {
public:
    OPENVINO_OP("FullyConnectedCompressed", "gpu_opset", FullyConnected);

    FullyConnectedCompressed() = default;

    FullyConnectedCompressed(const ov::Output<Node>& A,
                             const ov::Output<Node>& B,
                             const ov::Output<Node>& bias,
                             const ov::Output<Node>& w_decompression_scale,
                             const ov::Output<Node>& w_decompression_zero_point,
                             const ov::Output<Node>& a_decompression_scale,
                             const ov::Output<Node>& a_decompression_zero_point,
                             const ov::Output<Node>& a_precomputed_reduction,
                             const ov::element::Type output_type = ov::element::dynamic,
                             const bool transpose_b = true,
                             const int64_t scale_ifm_dim_idx = 1,
                             const int64_t zp_ifm_dim_idx = 1);

    FullyConnectedCompressed(const ov::Output<Node>& A,
                             const ov::Output<Node>& B,
                             const ov::Output<Node>& bias,
                             const ov::Output<Node>& w_decompression_scale,
                             const ov::Output<Node>& w_decompression_zero_point,
                             const ov::element::Type output_type = ov::element::dynamic,
                             const bool transpose_b = true,
                             const int64_t scale_ifm_dim_idx = 1,
                             const int64_t zp_ifm_dim_idx = 1);

    FullyConnectedCompressed(const ov::Output<Node>& A,
                             const ov::Output<Node>& B,
                             const ov::Output<Node>& bias,
                             const ov::Output<Node>& w_decompression_scale,
                             const ov::element::Type output_type = ov::element::dynamic,
                             const bool transpose_b = true,
                             const int64_t scale_ifm_dim_idx = 1,
                             const int64_t zp_ifm_dim_idx = 1);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    int64_t get_scale_ifm_dim_idx() const { return m_scale_ifm_dim_idx; }
    int64_t get_zp_ifm_dim_idx() const { return m_zp_ifm_dim_idx; }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

private:
    // transpose_b by default indicates ifm_dim_idx is in the last dimension
    int64_t m_scale_ifm_dim_idx = 1;
    int64_t m_zp_ifm_dim_idx = 1;
};

}   // namespace ov::intel_gpu::op
