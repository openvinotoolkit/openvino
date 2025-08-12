// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/vl_sdpa.hpp"

#include "augru_sequence_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"

namespace ov {
namespace op {
namespace internal {

VLSDPA::VLSDPA(const OutputVector& inputs,
               const std::vector<int64_t>& order_q,
               const std::vector<int64_t>& order_k,
               const std::vector<int64_t>& order_v,
               const std::vector<int64_t>& order_out)
    : Op(inputs),
      m_order_q(order_q),
      m_order_k(order_k),
      m_order_v(order_v),
      m_order_out(order_out) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> VLSDPA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_VLSDPA_clone_with_new_inputs);
    return std::make_shared<VLSDPA>(new_args, m_order_q, m_order_k, m_order_v, m_order_out);
}

bool VLSDPA::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_VLSDPA_visit_attributes);
    visitor.on_attribute("order_q", m_order_q);
    visitor.on_attribute("order_k", m_order_k);
    visitor.on_attribute("order_v", m_order_v);
    visitor.on_attribute("order_out", m_order_out);
    return true;
}

void VLSDPA::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_VLSDPA_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() == 4, "VLSDPA must have 4 inputs whereas it has ", get_input_size());

    auto out_type = get_input_element_type(0);

    const auto& cu_seqlens_type = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this,
                          cu_seqlens_type.is_integral() || cu_seqlens_type.is_dynamic(),
                          "The element type of cu_seqlens must be integral.");

    for (size_t i = 1; i < 3; i++) {
        const auto& element_type = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, element_type),
                              "Mixed input types of K/V are not supported.");
    }
    NODE_VALIDATION_CHECK(this,
                          out_type.is_real() || out_type.is_dynamic(),
                          "The element type of the input tensor must be a floating-point.");

    const auto& input_shapes = ov::util::get_node_input_partial_shapes(*this);

    // validate input shapes
    // VLSDPA node is only optimized by QWen2.x-VL model at the moment. Therefore,
    // the strict check is applied, which could be relaxed once we see similar patterns in
    // more models and corresponding kernel implements the function.
    auto shape_q = input_shapes[0];
    auto shape_k = input_shapes[1];
    auto shape_v = input_shapes[2];

    auto shape_q_rank = shape_q.rank();
    NODE_VALIDATION_CHECK(this,
                          shape_q_rank.is_static() && shape_q_rank.get_length() == 3,
                          "Query input rank length must be 3.");
    auto shape_v_rank = shape_v.rank();
    NODE_VALIDATION_CHECK(this,
                          shape_v_rank.is_static() && shape_v_rank.get_length() == 3,
                          "Key input rank length must be 3.");
    auto shape_k_rank = shape_v.rank();
    NODE_VALIDATION_CHECK(this,
                          shape_k_rank.is_static() && shape_k_rank.get_length() == 3,
                          "Value input rank length must be 3.");

    NODE_VALIDATION_CHECK(this,
                          (m_order_q == m_order_k && m_order_q == m_order_v && m_order_q == m_order_out),
                          "Value of m_order* must be equal.");

    if (m_order_q.size() > 0) {
        NODE_VALIDATION_CHECK(this,
                              (m_order_q == std::vector<int64_t>{1, 0, 2}),
                              "Value of order_q must be {1, 0, 2}.");
    }

    // const auto output_shapes = shape_infer(this, input_shapes);
    // transpose shape into BHLS(4D), or HLS(3D)
    auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
        if (order.empty())
            return pshape;

        auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
        for (size_t i = 0; i < order.size(); i++) {
            transposed_pshape[i] = pshape[order[i]];
        }
        return transposed_pshape;
    };
    const auto& output_shape = transpose_pshape(input_shapes[0], m_order_q);
    if (m_order_out.size() > 0) {
        set_output_type(0, out_type, transpose_pshape(output_shape, m_order_out));
    } else {
        set_output_type(0, out_type, output_shape);
    }
}

}  // namespace internal
}  // namespace op
}  // namespace ov
