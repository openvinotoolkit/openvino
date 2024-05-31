// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

IndirectSDPA::IndirectSDPA(const ov::Output<Node>& Q,
                           const ov::Output<Node>& K,
                           const ov::Output<Node>& V,
                           const ov::Output<Node>& beam_table,
                           const bool is_causal,
                           const int64_t indirect_axis,
                           const std::vector<int64_t>& order_q,
                           const std::vector<int64_t>& order_k,
                           const std::vector<int64_t>& order_v,
                           const std::vector<int64_t>& order_out,
                           const ov::element::Type output_type)
    : ov::intel_gpu::op::SDPA(Q, K, V, order_q, order_k, order_v, order_out, is_causal, output_type)
    , m_indirect_axis(indirect_axis) {
    set_argument(3, beam_table);
    validate_and_infer_types();
}

IndirectSDPA::IndirectSDPA(const ov::Output<Node>& Q,
                           const ov::Output<Node>& K,
                           const ov::Output<Node>& V,
                           const ov::Output<Node>& attn_mask,
                           const ov::Output<Node>& beam_table,
                           const bool is_causal,
                           const int64_t indirect_axis,
                           const std::vector<int64_t>& order_q,
                           const std::vector<int64_t>& order_k,
                           const std::vector<int64_t>& order_v,
                           const std::vector<int64_t>& order_out,
                           const ov::element::Type output_type)
    : ov::intel_gpu::op::SDPA(Q, K, V, attn_mask, order_q, order_k, order_v, order_out, is_causal, output_type)
    , m_indirect_axis(indirect_axis) {
    set_argument(4, beam_table);
    validate_and_infer_types();
}

IndirectSDPA::IndirectSDPA(const ov::Output<Node>& Q,
                           const ov::Output<Node>& K,
                           const ov::Output<Node>& V,
                           const ov::Output<Node>& attn_mask,
                           const ov::Output<Node>& scale,
                           const ov::Output<Node>& beam_table,
                           const bool is_causal,
                           const int64_t indirect_axis,
                           const std::vector<int64_t>& order_q,
                           const std::vector<int64_t>& order_k,
                           const std::vector<int64_t>& order_v,
                           const std::vector<int64_t>& order_out,
                           const ov::element::Type output_type)
    : ov::intel_gpu::op::SDPA(Q, K, V, attn_mask, scale, order_q, order_k, order_v, order_out, is_causal, output_type)
    , m_indirect_axis(indirect_axis) {
    set_argument(5, beam_table);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> IndirectSDPA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    if (new_args.size() == 4) {
        return std::make_shared<IndirectSDPA>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                              m_is_causal, m_indirect_axis, m_order_q, m_order_k, m_order_v, m_order_out, m_output_type);
    } else if (new_args.size() == 5) {
        return std::make_shared<IndirectSDPA>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4),
                                              m_is_causal, m_indirect_axis, m_order_q, m_order_k, m_order_v, m_order_out, m_output_type);
    } else {
        return std::make_shared<IndirectSDPA>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4), new_args.at(5),
                                              m_is_causal, m_indirect_axis, m_order_q, m_order_k, m_order_v, m_order_out, m_output_type);
    }
}

void IndirectSDPA::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 4 || input_size == 5 || input_size == 6,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 4, 5 or 6.");

    std::vector<ov::PartialShape> input_shapes;
    for (size_t i = 0; i < input_size - 1; i++) {
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

bool IndirectSDPA::visit_attributes(ov::AttributeVisitor &visitor) {
    SDPA::visit_attributes(visitor);
    visitor.on_attribute("indirect_axis", m_indirect_axis);
    return true;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
