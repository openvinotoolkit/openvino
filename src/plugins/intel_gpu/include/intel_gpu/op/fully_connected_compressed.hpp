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
                             const ov::Output<Node> &decompression_scale,
                             const ov::Output<Node> &decompression_zero_point,
                             const ov::element::Type output_type = ov::element::undefined);

    FullyConnectedCompressed(const ov::Output<Node> &A,
                             const ov::Output<Node> &B,
                             const ov::Output<Node> &bias,
                             const ov::Output<Node> &decompression_scale,
                             const ov::element::Type output_type = ov::element::undefined);

    FullyConnectedCompressed(const OutputVector& inputs,
                             bool has_zp = true,
                             bool has_activation_scale = false,
                             const ov::element::Type output_type = ov::element::undefined);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool get_has_zp() const { return m_has_zp; }
    bool get_has_activation_scale() const { return m_has_activation_scale; }


protected:
    bool m_has_zp;
    bool m_has_activation_scale;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
