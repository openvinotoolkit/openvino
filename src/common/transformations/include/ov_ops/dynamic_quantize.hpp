// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

struct QuantizationConfig {
    enum class QuantizationMode {
        Symmetric,
        Asymmetric
    };

    QuantizationMode mode = QuantizationMode::Symmetric;
    element::Type quantization_dt = element::undefined;
    element::Type scale_dt = element::undefined;
    element::Type zp_dt = element::undefined;
    std::vector<uint64_t> group_sizes = {};

    bool operator==(const QuantizationConfig& rhs) const {
        return mode == rhs.mode &&
               quantization_dt == rhs.quantization_dt &&
               scale_dt == rhs.scale_dt &&
               zp_dt == rhs.zp_dt &&
               group_sizes == rhs.group_sizes;
    }

    bool is_asymmetric_quantization() const {
        return mode == QuantizationMode::Asymmetric;
    }
};

/// \brief Operator performing Dynamic Quantize
class TRANSFORMATIONS_API DynamicQuantize : public ov::op::Op {
public:
    OPENVINO_OP("DynamicQuantize", "ie_internal_opset");
    DynamicQuantize() = default;
    /// \brief Constructs an DynamicQuantize operation.
    ///
    /// \param data Input tensor with data
    /// \param config Dynamic quantization configuration
    DynamicQuantize(const Output<Node>& data, const QuantizationConfig& config);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const std::vector<uint64_t>& get_group_sizes() const {
        return m_config.group_sizes;
    };

    QuantizationConfig::QuantizationMode get_quantization_mode() const {
        return m_config.mode;
    };

    QuantizationConfig get_quantization_config() const {
        return m_config;
    };

    static std::vector<ov::PartialShape> shape_infer(const DynamicQuantize* op,
                                                     const std::vector<ov::PartialShape>& input_shapes,
                                                     const QuantizationConfig& config);

protected:
    DynamicQuantize(const Output<Node>& data, const QuantizationConfig& config, size_t outputs_number);

    QuantizationConfig m_config;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
