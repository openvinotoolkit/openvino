// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief Operator that implements Key-Values cache subgraph for large language models.
/// This operation updates data of the corresponding Variable
class KVCache : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("KVCache", "gpu_opset");

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    KVCache() = default;

    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            const ov::element::Type output_type = ov::element::undefined);

    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const Output<Node>& beam_idx,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            int64_t gather_axis,
            const ov::element::Type output_type = ov::element::undefined);

    KVCache(const OutputVector& inputs,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            int64_t gather_axis,
            bool combine_scales_and_zp,
            const QuantizationConfig& config,
            const std::vector<uint64_t>& scales_zp_output_order,
            const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

    int64_t get_concat_axis() const { return m_concat_axis; }
    int64_t get_gather_axis() const { return m_gather_axis; }

    void set_concat_axis(int64_t axis) { m_concat_axis = axis; }
    void set_gather_axis(int64_t axis) { m_gather_axis = axis; }

    bool get_indirect() const { return m_indirect; }

    bool get_kv_compressed() const { return m_compressed; }
    bool get_combine_scales_and_zp() const { return m_combine_scales_and_zp; }
    QuantizationConfig get_quantization_config() const { return m_quantization_config; }
    std::vector<uint64_t> get_scales_zp_output_order() const { return m_scales_zp_output_order; }

private:
    int64_t m_concat_axis = 0;
    int64_t m_gather_axis = 0;
    bool m_indirect = false;

    bool m_compressed = false;
    bool m_combine_scales_and_zp = false;
    QuantizationConfig m_quantization_config = {};
    std::vector<uint64_t> m_scales_zp_output_order = {};

    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const KVCache* op, const std::vector<ov::PartialShape>& input_shapes);

std::vector<ov::PartialShape> shape_infer(const KVCache* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          const ov::op::internal::QuantizationConfig& config,
                                          const std::vector<uint64_t>& scales_output_order = {},
                                          bool combine_scales_and_zp = false);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
