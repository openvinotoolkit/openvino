// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

SDPA::SDPA(const OutputVector& inputs,
           const bool is_causal,
           const std::vector<int64_t>& order_q,
           const std::vector<int64_t>& order_k,
           const std::vector<int64_t>& order_v,
           const std::vector<int64_t>& order_out,
           const ov::element::Type output_type)
    : m_is_causal(is_causal)
    , m_order_q(order_q)
    , m_order_k(order_k)
    , m_order_v(order_v)
    , m_order_out(order_out)
    , m_output_type(output_type)
    , m_compressed(false) {
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
           const QuantizationConfig& quantization_config,
           const bool combine_scales_and_zp,
           const ov::element::Type output_type)
    : m_is_causal(is_causal)
    , m_order_q(order_q)
    , m_order_k(order_k)
    , m_order_v(order_v)
    , m_order_out(order_out)
    , m_output_type(output_type)
    , m_compressed(true)
    , m_combine_scales_and_zp(combine_scales_and_zp)
    , m_quantization_config(quantization_config) {
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
                                  m_output_type);
}

void SDPA::validate_and_infer_types() {
    const auto input_size = get_input_size();

    const auto compression_inputs = get_compression_inputs_num();
    NODE_VALIDATION_CHECK(this,
        input_size >= 3 + compression_inputs && input_size <= 5 + compression_inputs,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 3, 4 or 5 data inputs and ", compression_inputs, " KV-cache compression related inputs");

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

    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool SDPA::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("order_q", m_order_q);
    visitor.on_attribute("order_k", m_order_k);
    visitor.on_attribute("order_v", m_order_v);
    visitor.on_attribute("order_out", m_order_out);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

size_t SDPA::get_compression_inputs_num() const {
    size_t compression_inputs = 0;
    if (m_compressed) {
        compression_inputs += 2; // 2 * scales

        if (m_quantization_config.is_asymmetric_quantization() && !m_combine_scales_and_zp)
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

    const auto is_broadcastable = shape_k_t.rank().is_static() &&
                                  shape_v_t.rank().is_static() &&
                                  ((shape_q_t.size() == shape_k_t.size()) && (shape_q_t.size() == shape_v_t.size()));
    if (is_broadcastable) {
        size_t max_rank = shape_q_t.size();
        for (size_t i = 0; i < max_rank; ++i) {
            if (shape_q_t[i].is_static() && shape_k_t[i].is_static() && shape_v_t[i].is_static()) {
                auto broadcasted_dim = shape_q_t[i].get_length();
                shape_k_t[i] = broadcasted_dim;
                shape_v_t[i] = broadcasted_dim;
            }
        }
    }

    std::vector<ov::PartialShape> transposed_input_shapes{ shape_q_t, shape_k_t, shape_v_t };
    for (size_t i = 3; i < transposed_input_shapes.size(); i++) {
        transposed_input_shapes.push_back(input_shapes[i]);
    }

    OPENVINO_ASSERT(op != nullptr, "op should not be nullptr for shape_infer.");
    auto op_v13 = dynamic_cast<const ov::op::v13::ScaledDotProductAttention*>(op);
    OPENVINO_ASSERT(op_v13 != nullptr, "ov::op::v13::ScaledDotProductAttention*>(op) should not be nullptr.");
    auto out_shapes = ov::op::v13::shape_infer(op_v13, transposed_input_shapes);

    if (order_out.size() > 0) {
        return { transpose_pshape(out_shapes[0], order_out) };
    } else {
        return { out_shapes[0] };
    }
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
