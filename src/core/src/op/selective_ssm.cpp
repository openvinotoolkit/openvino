// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "openvino/util/common_util.hpp"
#include "selective_ssm_shape_inference.hpp"

namespace {

inline void selective_ssm_input_check(const ov::Node* node,
                                      size_t idx,
                                      const std::string_view input_name,
                                      std::initializer_list<ov::Rank>&& allowed_ranks,
                                      const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::element;
    using namespace ov::util;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& value) {
        return !value.is_dynamic() && is_rank_compatible_any_of(value.get_length(), allowed_ranks);
    };
    auto type_check = [&](const Type& value) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return !value.is_dynamic() && (allowed_types.empty() || it != allowed_types.end());
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

SelectiveSSM::SelectiveSSM(const Output<Node>& A,
                           const Output<Node>& dt,
                           const Output<Node>& B,
                           const Output<Node>& x,
                           const Output<Node>& C,
                           const Output<Node>& recurrent_state)
    : Op({A, dt, B, x, C, recurrent_state}) {
    constructor_validate_and_infer_types();
}

SelectiveSSM::SelectiveSSM(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void SelectiveSSM::validate_and_infer_types() {
    OV_OP_SCOPE(SelectiveSSM_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 6, "SelectiveSSM expects 6 inputs, but it has ", get_input_size());

    const std::vector<ov::element::Type> float_types{ov::element::f32, ov::element::f16, ov::element::bf16};
    selective_ssm_input_check(this, 0, "A", {1}, float_types);
    selective_ssm_input_check(this, 1, "dt", {3}, float_types);
    selective_ssm_input_check(this, 2, "B", {4}, float_types);
    selective_ssm_input_check(this, 3, "x", {4}, float_types);
    selective_ssm_input_check(this, 4, "C", {4}, float_types);
    selective_ssm_input_check(this, 5, "recurrent_state", {4}, float_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(3), output_shapes[0]);
    set_output_type(1, get_input_element_type(5), output_shapes[1]);
}

bool SelectiveSSM::visit_attributes(AttributeVisitor&) {
    OV_OP_SCOPE(SelectiveSSM_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> SelectiveSSM::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SelectiveSSM>(new_args);
}

}  // namespace ov::op::internal
