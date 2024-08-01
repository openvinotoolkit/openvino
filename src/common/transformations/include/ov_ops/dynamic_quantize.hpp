// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief Operator performing Dynamic Quantize
class TRANSFORMATIONS_API DynamicQuantize : public ov::op::Op {
public:
    OPENVINO_OP("DynamicQuantize", "gpu_opset");

    DynamicQuantize() = default;
    /// \brief Constructs an DynamicQuantize operation.
    ///
    /// \param data Input tensor with data
    /// \param group_size Group size for dynamic quantization
    /// \param dt_scale Data type for scale output
    DynamicQuantize(const Output<Node>& data, size_t group_size, element::Type dt_scale);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    size_t get_group_size() { return m_group_size; };
    static std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op, std::vector<ov::PartialShape> input_shapes);

private:
    size_t m_group_size;
    element::Type m_dt_scale;
};


}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
