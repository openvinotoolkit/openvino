// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reshape.hpp"

namespace ov::intel_gpu::op {

SDPA::SDPA(const OutputVector& inputs,
           const bool is_causal,
           const std::vector<int64_t>& order_q,
           const std::vector<int64_t>& order_k,
           const std::vector<int64_t>& order_v,
           const std::vector<int64_t>& order_out,
           const ov::element::Type output_type,
           bool split_kv)
    : m_is_causal(is_causal)
    , m_order_q(order_q)
    , m_order_k(order_k)
    , m_order_v(order_v)
    , m_order_out(order_out)
    , m_output_type(output_type)
    , m_compressed(false)
    , m_split_kv(split_kv) {
    set_arguments(inputs);
    set_causal(is_causal);
    validate_and_infer_types();
}

SDPA::SDPA(const OutputVector& inputs,
           const bool is_causal,
           const std::vector<int64_t>& order_q,
           const std::vector<int64_t>& order_k,
           const std::vector<int64_t>& order_v,
           const std::vector<int64_t>& order_out,
           const QuantizationAttribute& quantization_attrs,
           const ov::element::Type output_type)
    : m_is_causal(is_causal)
    , m_order_q(order_q)
    , m_order_k(order_k)
    , m_order_v(order_v)
    , m_order_out(order_out)
    , m_output_type(output_type)
    , m_compressed(true)
    , m_quantization_attrs(quantization_attrs) {
    set_arguments(inputs);
    set_causal(is_causal);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> SDPA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<SDPA>(new_args,
                                  m_is_causal,
                                  m_order_q,
                                  m_order_k,
                                  m_order_v,
                                  m_order_out,
                                  m_output_type,
                                  m_split_kv);
}

void SDPA::validate_and_infer_types() {
    const auto input_size = get_input_size();

    if (m_split_kv) {
        // split_kv layout: [Q, K_cache, V_cache, (mask), K_new, V_new, kv_len] -- 3 base inputs +
        // optional mask + 2 trailing K_new/V_new + the trailing i32 kv_len (6 or 7 total). No scale
        // INPUT (scale is the scale_val attribute). KV-cache compression / indirect / sink are not
        // supported together with split_kv.
        NODE_VALIDATION_CHECK(this,
            !m_compressed,
            "split_kv SDPA does not support KV-cache compression.");
        NODE_VALIDATION_CHECK(this,
            input_size == 6 || input_size == 7,
            "Number of inputs is incorrect for split_kv SDPA. Current value is: ",
            input_size,
            ", expected [Q, K_cache, V_cache, (mask), K_new, V_new, kv_len]");
    } else {
        const auto compression_inputs = get_compression_inputs_num();
        NODE_VALIDATION_CHECK(this,
            input_size >= 3 + compression_inputs && input_size <= 5 + compression_inputs,
            "Number of inputs is incorrect. Current value is: ",
            input_size,
            ", expected 3, 4 or 5 data inputs and ", compression_inputs, " KV-cache compression related inputs");
    }

    std::vector<ov::PartialShape> input_shapes;
    for (size_t i = 0; i < input_size; i++) {
        input_shapes.push_back(get_input_partial_shape(i));
    }

    auto out_shapes = shape_infer(this,
                                  input_shapes,
                                  m_order_q,
                                  m_order_k,
                                  m_order_v,
                                  m_order_out);

    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool SDPA::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("order_q", m_order_q);
    visitor.on_attribute("order_k", m_order_k);
    visitor.on_attribute("order_v", m_order_v);
    visitor.on_attribute("order_out", m_order_out);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("split_kv", m_split_kv);
    return true;
}

size_t SDPA::get_compression_inputs_num() const {
    size_t compression_inputs = 0;
    if (m_compressed) {
        compression_inputs += 2; // 2 * scales

        if (m_quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
            m_quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar)
            compression_inputs += 2; // 2 * zp
    }

    return compression_inputs;
}

std::vector<ov::PartialShape> shape_infer(const SDPA* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_q,
                                          const std::vector<int64_t>& order_k,
                                          const std::vector<int64_t>& order_v,
                                          const std::vector<int64_t>& order_out) {
    auto shape_q = input_shapes[0];
    auto shape_k = input_shapes[1];
    auto shape_v = input_shapes[2];

    // transposed shape
    auto transpose_pshape = [](const ov::PartialShape pshape, const std::vector<int64_t>& order) {
        auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
        for (size_t i = 0; i < order.size(); i++) {
            transposed_pshape[i] = pshape[order[i]];
        }

        return transposed_pshape;
    };

    auto shape_q_t = (order_q.size() > 1) ? transpose_pshape(shape_q, order_q) : shape_q;
    auto shape_k_t = (order_k.size() > 1) ? transpose_pshape(shape_k, order_k) : shape_k;
    auto shape_v_t = (order_v.size() > 1) ? transpose_pshape(shape_v, order_v) : shape_v;

    // split_kv: K and V are supplied as two chunks each. input_shapes is
    // [Q, K_cache, V_cache, (mask), K_new, V_new] (K_new/V_new are always the last two inputs);
    // the new chunks reuse K/V's transpose order. The op attends over the logical sequence
    // concatenation, so the effective K/V seq length is (cache_seq + new_seq). Fold the new chunk's
    // seq dim into the cache shape here so the standard v13 inference below produces the
    // concatenated output shape. In transposed [B,H,S,D] space the sequence axis is the
    // second-to-last dimension.
    if (op != nullptr && op->get_split_kv() && shape_k_t.rank().is_static() && shape_v_t.rank().is_static() &&
        shape_k_t.size() >= 2 && shape_v_t.size() >= 2 && input_shapes.size() >= 3) {
        // Inputs end with [.., K_new, V_new, kv_len], so K_new/V_new are the 3rd/2nd from last
        // (the trailing slot is the i32 kv_len control tensor, not a K/V chunk).
        const auto shape_k_new = input_shapes[input_shapes.size() - 3];
        const auto shape_v_new = input_shapes[input_shapes.size() - 2];
        const auto shape_k_new_t = (order_k.size() > 1) ? transpose_pshape(shape_k_new, order_k) : shape_k_new;
        const auto shape_v_new_t = (order_v.size() > 1) ? transpose_pshape(shape_v_new, order_v) : shape_v_new;
        const auto k_seq_axis = shape_k_t.size() - 2;
        const auto v_seq_axis = shape_v_t.size() - 2;
        if (shape_k_new_t.rank().is_static() && shape_k_new_t.size() == shape_k_t.size()) {
            shape_k_t[k_seq_axis] += shape_k_new_t[k_seq_axis];
        }
        if (shape_v_new_t.rank().is_static() && shape_v_new_t.size() == shape_v_t.size()) {
            shape_v_t[v_seq_axis] += shape_v_new_t[v_seq_axis];
        }
    }

    const auto is_broadcastable = shape_k_t.rank().is_static() &&
                                  shape_v_t.rank().is_static() &&
                                  ((shape_q_t.size() == shape_k_t.size()) && (shape_q_t.size() == shape_v_t.size()));
    if (is_broadcastable) {
        size_t max_rank = shape_q_t.size() -1;
        for (size_t i = 0; i < max_rank; ++i) {
            if (shape_q_t[i].is_static() && shape_k_t[i].is_static()) {
                auto broadcasted_dim = shape_q_t[i].get_length();
                shape_k_t[i] = broadcasted_dim;
            }

            if (shape_q_t[i].is_static() && shape_v_t[i].is_static()) {
                auto broadcasted_dim = shape_q_t[i].get_length();
                shape_v_t[i] = broadcasted_dim;
            }
        }
    }

    std::vector<ov::PartialShape> transposed_input_shapes{ shape_q_t, shape_k_t, shape_v_t };
    for (size_t i = 3; i < transposed_input_shapes.size(); i++) {
        transposed_input_shapes.push_back(input_shapes[i]);
    }

    OPENVINO_ASSERT(op != nullptr, "op should not be nullptr for shape_infer.");
    auto op_v13 = ov::as_type<const ov::op::v13::ScaledDotProductAttention>(op);
    OPENVINO_ASSERT(op_v13 != nullptr, "ov::op::v13::ScaledDotProductAttention*>(op) should not be nullptr.");
    auto out_shapes = ov::op::v13::shape_infer(op_v13, transposed_input_shapes);

    if (order_out.size() > 0) {
        return { transpose_pshape(out_shapes[0], order_out) };
    } else {
        return { out_shapes[0] };
    }
}

}  // namespace ov::intel_gpu::op
