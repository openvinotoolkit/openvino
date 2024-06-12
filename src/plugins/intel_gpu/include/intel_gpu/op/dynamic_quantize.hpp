// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief Operator performing Dynamic Quantize
class DynamicQuantize : public ov::op::Op {
public:
    OPENVINO_OP("DynamicQuantize", "gpu_opset");

    DynamicQuantize() = default;
    /// \brief Constructs an DynamicQuantize operation.
    ///
    /// \param data Input tensor with data
    DynamicQuantize(const Output<Node>& data);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

private:
};

std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op, std::vector<ov::PartialShape> input_shapes);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
