// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <algorithm>
#include <string_view>

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_causal_conv1d_shape_inference.hpp"

namespace {

// Validates input rank and type for a node input.
inline void paged_causal_conv1d_input_check(const ov::Node* node,
                                            size_t idx,
                                            const std::string_view input_name,
                                            const std::initializer_list<ov::Rank>& allowed_ranks,
                                            const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), type);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input must be one of [",
                          join(allowed_ranks),
                          "]. Got: ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input must be one of [",
                          join(allowed_types),
                          "]. Got: ",
                          tp,
                          ".");
}
}  // namespace

namespace ov::op::internal {

PagedCausalConv1D::PagedCausalConv1D(const Output<Node>& input_embeds,
                                     const Output<Node>& conv_state_table,
                                     const Output<Node>& conv_weight,
                                     const Output<Node>& conv_bias,
                                     const Output<Node>& subsequence_begins,
                                     const Output<Node>& la_block_indices,
                                     const Output<Node>& la_block_indices_begins,
                                     const Output<Node>& processed_tokens,
                                     const Output<Node>& cache_interval)
    : Op({input_embeds,
          conv_state_table,
          conv_weight,
          conv_bias,
          subsequence_begins,
          la_block_indices,
          la_block_indices_begins,
          processed_tokens,
          cache_interval}) {
    constructor_validate_and_infer_types();
}

PagedCausalConv1D::PagedCausalConv1D(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void PagedCausalConv1D::validate_and_infer_types() {
    OV_OP_SCOPE(PagedCausalConv1D_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 9, "PagedCausalConv1D expects 9 inputs. Got: ", get_input_size());

    static const std::vector<ov::element::Type> float_types = {ov::element::f32, ov::element::f16, ov::element::bf16};
    static const std::vector<ov::element::Type> integer_types = {ov::element::i32, ov::element::i64};

    paged_causal_conv1d_input_check(this, 0, "input_embeds", {2}, float_types);
    paged_causal_conv1d_input_check(this, 1, "conv_state_table", {3}, float_types);
    paged_causal_conv1d_input_check(this, 2, "conv_weight", {3}, float_types);
    paged_causal_conv1d_input_check(this, 3, "conv_bias", {1}, float_types);
    paged_causal_conv1d_input_check(this, 4, "subsequence_begins", {1}, integer_types);
    paged_causal_conv1d_input_check(this, 5, "la_block_indices", {1}, integer_types);
    paged_causal_conv1d_input_check(this, 6, "la_block_indices_begins", {1}, integer_types);
    paged_causal_conv1d_input_check(this, 7, "processed_tokens", {1}, integer_types);
    paged_causal_conv1d_input_check(this, 8, "cache_interval", {1}, integer_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool PagedCausalConv1D::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedCausalConv1D_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> PagedCausalConv1D::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedCausalConv1D_clone_with_new_inputs);
    return std::make_shared<PagedCausalConv1D>(new_args);
}

}  // namespace ov::op::internal
