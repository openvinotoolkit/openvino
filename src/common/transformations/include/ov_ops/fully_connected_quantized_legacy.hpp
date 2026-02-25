// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API FullyConnectedQuantizedLegacy : public FullyConnected {
public:
    OPENVINO_OP("FullyConnectedQuantizedLegacy", "ie_internal_opset", FullyConnected);

    FullyConnectedQuantizedLegacy() = default;

    FullyConnectedQuantizedLegacy(const ov::Output<Node>& X,
                                  const ov::Output<Node>& W,
                                  const ov::Output<Node>& bias,
                                  const ov::Output<Node>& deq_scales,
                                  const ov::Output<Node>& deq_zero_points,
                                  const ov::element::Type output_type = ov::element::dynamic);

    FullyConnectedQuantizedLegacy(const ov::Output<Node>& X,
                                  const ov::Output<Node>& W,
                                  const ov::Output<Node>& bias,
                                  const ov::Output<Node>& deq_scales,
                                  const ov::element::Type output_type = ov::element::dynamic);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
