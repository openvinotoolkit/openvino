// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/kv_cache.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace ov::intel_gpu::op {

/// \brief Operator that implements Key-Values cache subgraph for large language models.
/// This operation updates data of the corresponding Variable
class KVCacheCompressed : public ov::intel_gpu::op::KVCache {
public:
    OPENVINO_OP("KVCacheCompressed", "gpu_opset", ov::intel_gpu::op::KVCache);

    using QuantizationAttrs = ov::op::internal::DynamicQuantize::Attributes;

    KVCacheCompressed() = default;

    KVCacheCompressed(const OutputVector& inputs,
                      const std::shared_ptr<ov::op::util::Variable>& past_values,
                      int64_t concat_axis,
                      int64_t gather_axis,
                      const QuantizationAttrs& quantization_attrs,
                      const ov::element::Type output_type = ov::element::dynamic);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool get_kv_compressed() const { return m_compressed; }
    bool get_combine_scales_and_zp() const {
        return m_quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
               m_quantization_attrs.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
    }

    QuantizationAttrs get_quantization_attrs() const { return m_quantization_attrs; }
    void set_quantization_attrs(QuantizationAttrs attrs) { m_quantization_attrs = std::move(attrs); }

    std::vector<uint64_t> get_scales_zp_output_order() const { return m_quantization_attrs.scales_zp_output_order; }

private:
    bool m_compressed;
    QuantizationAttrs m_quantization_attrs = {};
};

std::vector<ov::PartialShape> shape_infer(const KVCacheCompressed* op,
                                          const std::vector<ov::PartialShape>& input_shapes);

}   // namespace ov::intel_gpu::op
