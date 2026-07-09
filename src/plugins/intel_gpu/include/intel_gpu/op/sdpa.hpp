// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace ov::intel_gpu::op {

class SDPA : public ov::op::v13::ScaledDotProductAttention {
public:
    OPENVINO_OP("SDPA", "gpu_opset", ov::op::v13::ScaledDotProductAttention);

    using QuantizationAttribute = ov::op::internal::DynamicQuantize::Attributes;

    SDPA() = default;

    // split_kv: when true, the input list carries K and V as two pieces each, in the fixed order
    // [Q, K_cache, V_cache, mask, K_new, V_new, kv_len]. There is no scale input (scale is the
    // scale_val attribute, fixed at 1.0 by the fusion). The trailing i32 kv_len is the valid cache
    // length; the kernel caps its cache loops there. The op attends over the logical sequence
    // concatenation of the cache and new chunks without materializing it, avoiding a per-step
    // full-cache KV Concat. Defaults to false, leaving the standard single-K/single-V op unchanged.
    SDPA(const OutputVector& inputs,
         const bool is_causal,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const ov::element::Type output_type = ov::element::dynamic,
         bool split_kv = false);

    SDPA(const OutputVector& inputs,
         const bool is_causal,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const QuantizationAttribute& quantization_attrs,
         const ov::element::Type output_type = ov::element::dynamic);

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
    QuantizationAttribute get_quantization_attrs() const { return m_quantization_attrs; }
    size_t get_compression_inputs_num() const;

    // True when K/V are supplied as two chunks each (cache + new); see the split_kv constructor.
    bool get_split_kv() const { return m_split_kv; }

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
    QuantizationAttribute m_quantization_attrs = {};
    bool m_split_kv = false;
};

std::vector<ov::PartialShape> shape_infer(const SDPA* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_q,
                                          const std::vector<int64_t>& order_k,
                                          const std::vector<int64_t>& order_v,
                                          const std::vector<int64_t>& order_out);


}   // namespace ov::intel_gpu::op
