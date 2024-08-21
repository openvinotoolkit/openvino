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

class TRANSFORMATIONS_API FullyConnectedQuantizedLegacy : public ov::op::internal::FullyConnected {
public:
    OPENVINO_OP("FullyConnectedQuantizedLegacy", "gpu_opset");

    FullyConnectedQuantizedLegacy() = default;

    FullyConnectedQuantizedLegacy(const OutputVector& arguments,
                                  const ov::element::Type output_type = ov::element::undefined);

    FullyConnectedQuantizedLegacy(const ov::Output<Node>& X,
                                  const ov::Output<Node>& W,
                                  const ov::Output<Node>& bias,
                                  const ov::Output<Node>& deq_scales,
                                  const ov::Output<Node>& deq_zero_points,
                                  const ov::element::Type output_type = ov::element::undefined);

    FullyConnectedQuantizedLegacy(const ov::Output<Node>& X,
                                  const ov::Output<Node>& W,
                                  const ov::Output<Node>& bias,
                                  const ov::Output<Node>& deq_scales,
                                  const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::shared_ptr<Node> fuse_bias(const ov::Output<Node>& bias) const override final;

    ov::element::Type get_output_type() const {
        return m_output_type;
    }
};

}  // namespace internal
}  // namespace op
}  // namespace ov
