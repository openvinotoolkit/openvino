// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class FullyConnectedCompressed : public FullyConnected {
public:
    OPENVINO_OP("FullyConnectedCompressed", "gpu_opset");

    FullyConnectedCompressed() = default;

    FullyConnectedCompressed(const ov::Output<Node> &A,
                             const ov::Output<Node> &B,
                             const ov::Output<Node> &bias,
                             const ov::Output<Node> &w_decompression_scale,
                             const ov::Output<Node> &w_decompression_zero_point,
                             const ov::Output<Node> &a_decompression_scale,
                             const ov::element::Type output_type = ov::element::undefined);


    FullyConnectedCompressed(const ov::Output<Node> &A,
                             const ov::Output<Node> &B,
                             const ov::Output<Node> &bias,
                             const ov::Output<Node> &w_decompression_scale,
                             const ov::Output<Node> &w_decompression_zero_point,
                             const ov::element::Type output_type = ov::element::undefined);

    FullyConnectedCompressed(const ov::Output<Node> &A,
                             const ov::Output<Node> &B,
                             const ov::Output<Node> &bias,
                             const ov::Output<Node> &w_decompression_scale,
                             const ov::element::Type output_type = ov::element::undefined);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
