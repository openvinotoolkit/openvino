// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include "dimension_util.hpp"
#include "gated_delta_net_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace {

// Validates input rank and type for a node input.
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return !rank.is_dynamic() && is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return !type.is_dynamic() && (allowed_types.empty() || it != allowed_types.end());
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}
}  // namespace

namespace ov::op::internal {

GatedDeltaNet::GatedDeltaNet(const Output<Node>& query,
                             const Output<Node>& key,
                             const Output<Node>& value,
                             const Output<Node>& recurrent_state,
                             const Output<Node>& gate,
                             const Output<Node>& beta)
    : Op({query, key, value, recurrent_state, gate, beta}) {
    constructor_validate_and_infer_types();
}

GatedDeltaNet::GatedDeltaNet(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void GatedDeltaNet::validate_and_infer_types() {
    OV_OP_SCOPE(GatedDeltaNet_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 6, "GatedDeltaNet expects 6 inputs, but it has ", get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    input_check(this, 0, "query", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 1, "key", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 2, "value", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 3, "recurrent_state", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 4, "gate", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 5, "beta", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, get_input_element_type(3), output_shapes[1]);
}

bool GatedDeltaNet::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GatedDeltaNet_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("fuse_qk_l2norm", m_config.fuse_qk_l2norm);
    visitor.on_attribute("q_l2_norm_eps", m_config.q_l2_norm_eps);
    visitor.on_attribute("k_l2_norm_eps", m_config.k_l2_norm_eps);
    visitor.finish_structure();
    return true;
}

std::shared_ptr<ov::Node> GatedDeltaNet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    auto cloned = std::make_shared<GatedDeltaNet>(new_args);
    cloned->m_config = m_config;
    return cloned;
}

}  // namespace ov::op::internal
