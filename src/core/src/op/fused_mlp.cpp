// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "openvino/op/fused_mlp.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {

namespace {

static void validate_rank_2_or_4(const ov::Node* node,
                                const ov::PartialShape& pshape,
                                const char* input_name) {
    NODE_VALIDATION_CHECK(node, pshape.rank().is_static(), input_name, " rank must be static");
    const auto r = pshape.rank().get_length();
    NODE_VALIDATION_CHECK(node, r == 2 || r == 4, input_name, " rank must be 2 or 4, got ", r);
}

static void validate_last_dims_are_one(const ov::Node* node,
                                      const ov::PartialShape& pshape,
                                      const char* input_name,
                                      size_t begin_idx) {
    NODE_VALIDATION_CHECK(node, pshape.rank().is_static(), input_name, " rank must be static");
    for (size_t i = begin_idx; i < pshape.size(); ++i) {
        NODE_VALIDATION_CHECK(node, pshape[i].is_static() && pshape[i].get_length() == 1,
                              input_name, " dimension ", i, " must be 1");
    }
}

static int64_t get_static_dim(const ov::Node* node, const ov::Dimension& d, const char* what) {
    NODE_VALIDATION_CHECK(node, d.is_static(), what, " must be static");
    return d.get_length();
}

}  // namespace

FusedMLP::FusedMLP(
    const ov::Output<Node>& x,
    const ov::Output<Node>& w_gate,
    const ov::Output<Node>& w_up,
    const ov::Output<Node>& w_down) : ov::op::Op({x, w_gate, w_up, w_down}) {
    constructor_validate_and_infer_types();
}

bool FusedMLP::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_FusedMLP_visit_attributes);
    return true;
}

void FusedMLP::validate_and_infer_types() {
    OV_OP_SCOPE(internal_FusedMLP_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "Number of inputs is incorrect. Expected 4, got ", get_input_size());

    const auto et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this, et == ov::element::f16, "Only f16 is supported for input X, got ", et);
    for (size_t i = 1; i < 4; ++i) {
        const auto w_et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this, w_et == ov::element::f16, "Only f16 is supported for weights, got ", w_et, " at input ", i);
    }

    const auto x_pshape = get_input_partial_shape(0);
    const auto w_gate_pshape = get_input_partial_shape(1);
    const auto w_up_pshape = get_input_partial_shape(2);
    const auto w_down_pshape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this, w_gate_pshape.is_static() && w_up_pshape.is_static() && w_down_pshape.is_static(),
                          "Only static shapes are supported (POC)");

    validate_rank_2_or_4(this, x_pshape, "X");
    validate_rank_2_or_4(this, w_gate_pshape, "W_gate");
    validate_rank_2_or_4(this, w_up_pshape, "W_up");
    validate_rank_2_or_4(this, w_down_pshape, "W_down");

    const auto x_rank = x_pshape.rank().get_length();
    const auto w_gate_rank = w_gate_pshape.rank().get_length();
    const auto w_up_rank = w_up_pshape.rank().get_length();
    const auto w_down_rank = w_down_pshape.rank().get_length();

    if (x_rank == 4) {
        validate_last_dims_are_one(this, x_pshape, "X", 3);
    }
    if (w_gate_rank == 4) {
        validate_last_dims_are_one(this, w_gate_pshape, "W_gate", 2);
    }
    if (w_up_rank == 4) {
        validate_last_dims_are_one(this, w_up_pshape, "W_up", 2);
    }
    if (w_down_rank == 4) {
        validate_last_dims_are_one(this, w_down_pshape, "W_down", 2);
    }

    const int64_t ic = (x_rank == 2) ? get_static_dim(this, x_pshape[1], "X[1]") : get_static_dim(this, x_pshape[2], "X[2]");

    const int64_t w_gate_ic = get_static_dim(this, w_gate_pshape[0], "W_gate[0]");
    const int64_t w_gate_oc = get_static_dim(this, w_gate_pshape[1], "W_gate[1]");
    NODE_VALIDATION_CHECK(this, w_gate_ic == ic, "W_gate[0] must equal IC (", ic, "), got ", w_gate_ic);

    const int64_t w_up_ic = get_static_dim(this, w_up_pshape[0], "W_up[0]");
    const int64_t w_up_oc = get_static_dim(this, w_up_pshape[1], "W_up[1]");
    NODE_VALIDATION_CHECK(this, w_up_ic == ic, "W_up[0] must equal IC (", ic, "), got ", w_up_ic);
    NODE_VALIDATION_CHECK(this, w_up_oc == w_gate_oc, "W_up[1] must equal W_gate[1] (", w_gate_oc, "), got ", w_up_oc);

    const int64_t w_down_oc = get_static_dim(this, w_down_pshape[0], "W_down[0]");
    const int64_t w_down_ic = get_static_dim(this, w_down_pshape[1], "W_down[1]");
    NODE_VALIDATION_CHECK(this, w_down_oc == w_gate_oc, "W_down[0] must equal OC (", w_gate_oc, "), got ", w_down_oc);
    NODE_VALIDATION_CHECK(this, w_down_ic == ic, "W_down[1] must equal IC (", ic, "), got ", w_down_ic);

    set_output_type(0, et, x_pshape);
}

std::shared_ptr<Node> FusedMLP::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_FusedMLP_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<FusedMLP>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

}  // namespace internal
}  // namespace op
}  // namespace ov
