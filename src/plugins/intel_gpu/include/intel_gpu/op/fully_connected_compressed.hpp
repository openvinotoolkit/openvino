// Copyright (C) 2018-2025 Intel Corporation
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
                             const ov::element::Type output_type = ov::element::dynamic);

    FullyConnectedCompressed(const ov::Output<Node>& A,
                             const ov::Output<Node>& B,
                             const ov::Output<Node>& bias,
                             const ov::Output<Node>& w_decompression_scale,
                             const ov::Output<Node>& w_decompression_zero_point,
                             const ov::element::Type output_type = ov::element::dynamic);

    FullyConnectedCompressed(const ov::Output<Node>& A,
                             const ov::Output<Node>& B,
                             const ov::Output<Node>& bias,
                             const ov::Output<Node>& w_decompression_scale,
                             const ov::element::Type output_type = ov::element::dynamic);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}   // namespace ov::intel_gpu::op
