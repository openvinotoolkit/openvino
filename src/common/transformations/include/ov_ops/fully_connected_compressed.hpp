// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API FullyConnectedCompressed : public FullyConnected {
public:
    OPENVINO_OP("FullyConnectedCompressed", "ie_internal_opset", FullyConnected);

    FullyConnectedCompressed() = default;

    FullyConnectedCompressed(const OutputVector& arguments,
                             const ov::element::Type output_type = ov::element::undefined);

    FullyConnectedCompressed(const ov::Output<Node>& X,
                             const ov::Output<Node>& W,
                             const ov::Output<Node>& bias,
                             const ov::Output<Node>& weight_scales,
                             const ov::Output<Node>& weight_zero_points,
                             const ov::element::Type output_type = ov::element::undefined);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::shared_ptr<Node> fuse_bias(const ov::Output<Node>& bias) const override final;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
