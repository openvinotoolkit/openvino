// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class SDPA : public ov::op::v13::ScaledDotProductAttention {
public:
    OPENVINO_OP("SDPA", "gpu_opset");

    using QuantizationConfig = ov::op::internal::QuantizationConfig;

    SDPA() = default;

    SDPA(const OutputVector& inputs,
         const bool is_causal,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const ov::element::Type output_type = ov::element::undefined);

    SDPA(const OutputVector& inputs,
         const bool is_causal,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const QuantizationConfig& quantization_config,
         const bool m_combine_scales_and_zp,
         const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool get_causal() const { return m_is_causal; }

    std::vector<int64_t> get_input0_transpose_order() const { return m_order_q; }
    std::vector<int64_t> get_input1_transpose_order() const { return m_order_k; }
    std::vector<int64_t> get_input2_transpose_order() const { return m_order_v; }
    std::vector<int64_t> get_output_transpose_order() const { return m_order_out; }
    ov::element::Type get_output_type() const { return m_output_type; }

    bool get_kv_compressed() const { return m_compressed; }
    bool get_combine_scales_and_zp() const { return m_combine_scales_and_zp; }
    QuantizationConfig get_quantization_config() const { return m_quantization_config; }
    size_t get_compression_inputs_num() const;

    static std::vector<int64_t> default_order(size_t rank) {
        std::vector<int64_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

protected:
    bool m_is_causal;
    std::vector<int64_t> m_order_q;
    std::vector<int64_t> m_order_k;
    std::vector<int64_t> m_order_v;
    std::vector<int64_t> m_order_out;
    ov::element::Type m_output_type;

    bool m_compressed = false;
    bool m_combine_scales_and_zp = false;
    QuantizationConfig m_quantization_config = {};
};

std::vector<ov::PartialShape> shape_infer(const SDPA* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_q,
                                          const std::vector<int64_t>& order_k,
                                          const std::vector<int64_t>& order_v,
                                          const std::vector<int64_t>& order_out);


}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
