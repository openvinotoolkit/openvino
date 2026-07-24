// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mamba2.hpp"

#include "itt.hpp"
#include "mamba2_shape_inference.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "openvino/util/common_util.hpp"

namespace {

// Validates input rank and type for a Mamba2 node input.
inline void mamba2_input_check(const ov::Node* node,
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

Mamba2::Mamba2(const Output<Node>& dA,
               const Output<Node>& dBx,
               const Output<Node>& C,
               const Output<Node>& recurrent_state)
    : Op({dA, dBx, C, recurrent_state}) {
    constructor_validate_and_infer_types();
}

Mamba2::Mamba2(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void Mamba2::validate_and_infer_types() {
    OV_OP_SCOPE(Mamba2_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "Mamba2 expects 4 inputs, but it has ", get_input_size());

    const std::vector<ov::element::Type> float_types{ov::element::f32, ov::element::f16, ov::element::bf16};
    // format: Node*, input_idx, name, {rank_list}, {type_list}
    mamba2_input_check(this, 0, "dA", {5}, float_types);
    mamba2_input_check(this, 1, "dBx", {5}, float_types);
    mamba2_input_check(this, 2, "C", {4}, float_types);
    mamba2_input_check(this, 3, "recurrent_state", {4}, float_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    // output has the per-token data type derived from `dBx`; output state keeps the recurrent_state type
    set_output_type(0, get_input_element_type(1), output_shapes[0]);
    set_output_type(1, get_input_element_type(3), output_shapes[1]);
}

std::shared_ptr<ov::Node> Mamba2::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<Mamba2>(new_args);
}

}  // namespace ov::op::internal
