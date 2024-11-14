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
    OPENVINO_OP("DynamicQuantize", "ie_internal_opset");

    /**
     * @brief Configuration for the type of quantization applied to the data:
     * - Symmetric: Quantization where the zero point is fixed at zero, and the range is symmetric around zero.
     * - Asymmetric: Quantization where the zero point is not fixed at zero.
     */
    enum class QuantizationType { Symmetric, Asymmetric };

    /**
     * @brief Configuration for how Activations, Scales and Zero Points will be stored in output buffers:
     * - Planar: Activations, Scales, and Zero Points are stored in independent buffers.
     * - InterleavedScalesZP: Activations are stored in an independent buffer, while Scales and Zero Points (if any) are
     *   combined in a separate buffer.
     */
    enum class OutputStorageType { Planar, InterleavedScalesZP, /* InterleavedActivationsScalesZP */ };

    /// \brief Structure that specifies attributes for interpolation
    struct Attributes {
        QuantizationType quantization_type = QuantizationType::Symmetric;
        element::Type quantization_dt = element::undefined;
        element::Type scale_dt = element::undefined;
        element::Type zp_dt = element::undefined;

        std::vector<uint64_t> group_sizes = {};
        std::vector<uint64_t> scales_zp_output_order = {};
        OutputStorageType output_storage_type = OutputStorageType::Planar;
    };

    DynamicQuantize() = default;
    /// \brief Constructs an DynamicQuantize operation.
    ///
    /// \param data Input tensor with data
    /// \param config Dynamic quantization configuration
    DynamicQuantize(const Output<Node>& data, const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }

    void set_attrs(Attributes attrs) {
        m_attrs = std::move(attrs);
    }

    const std::vector<uint64_t>& get_group_sizes() const {
        return m_attrs.group_sizes;
    }

    QuantizationType get_quantization_type() const {
        return m_attrs.quantization_type;
    }

    OutputStorageType get_output_storage_type() const {
        return m_attrs.output_storage_type;
    }

    const std::vector<uint64_t>& get_scales_zp_output_order() const {
        return m_attrs.scales_zp_output_order;
    }

    static std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op,
                                                     const std::vector<ov::PartialShape>& input_shapes);

protected:
    Attributes m_attrs;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
