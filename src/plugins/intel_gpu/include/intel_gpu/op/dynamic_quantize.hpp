// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class DynamicQuantize : public ov::op::internal::DynamicQuantize {
public:
    OPENVINO_OP("DynamicQuantize", "gpu_opset");

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    DynamicQuantize() = default;
    /// \brief Constructs an DynamicQuantize operation.
    ///
    /// \param data Input tensor with data
    /// \param config Dynamic quantization configuration
    /// \param scales_zp_output_order Specifies on default order of scales and zero points
    /// \param combine_scales_and_zp If true, combines scales and zero points into a single buffer, pairing each scale with its corresponding zero point
    DynamicQuantize(const Output<Node>& data,
                    const QuantizationConfig& config,
                    const std::vector<uint64_t>& scales_zp_output_order = {},
                    const bool combine_scales_and_zp = false);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const std::vector<uint64_t>& get_scales_zp_output_order() const {
        return m_scales_zp_output_order;
    }

    bool get_combine_scales_and_zp() const {
        return m_combine_scales_and_zp;
    }

    static std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op,
                                                     const std::vector<ov::PartialShape>& input_shapes,
                                                     const QuantizationConfig& config,
                                                     const std::vector<uint64_t>& scales_zp_output_order,
                                                     const bool combine_scales_and_zp = false);

private:
    bool m_combine_scales_and_zp = false;
    std::vector<uint64_t> m_scales_zp_output_order;
};

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
