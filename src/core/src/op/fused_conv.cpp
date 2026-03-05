// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fused_conv.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {

FusedConv::FusedConv(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void FusedConv::validate_and_infer_types() {
    OV_OP_SCOPE(FusedConv_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 4,
                          "FusedConv expects 4 inputs, but it has ",
                          get_input_size());

    // input[0]: [B, conv_dim, S]
    const auto& input_rank = get_input_partial_shape(0).rank();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() == 3,
                          "Rank of `input` should be 3, but it is ",
                          input_rank);

    // input[1]: [conv_dim, kernel_size]
    const auto& weight_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          weight_rank.is_dynamic() || weight_rank.get_length() == 2,
                          "Rank of `conv_weight` should be 2, but it is ",
                          weight_rank);

    // input[2]: [B] (i32/i64)
    const auto& beam_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          beam_rank.is_dynamic() || beam_rank.get_length() == 1,
                          "Rank of `beam_idx` should be 1, but it is ",
                          beam_rank);

    // input[3]: [B, conv_dim, kernel_size]
    const auto& state_rank = get_input_partial_shape(3).rank();
    NODE_VALIDATION_CHECK(this,
                          state_rank.is_dynamic() || state_rank.get_length() == 3,
                          "Rank of `initial_state` should be 3, but it is ",
                          state_rank);

    // output[0]: same shape as input[0] = [B, conv_dim, S]
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    // output[1]: same shape as input[3] = [B, conv_dim, kernel_size]
    set_output_type(1, get_input_element_type(3), get_input_partial_shape(3));
}

std::shared_ptr<ov::Node> FusedConv::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<FusedConv>(new_args);
}

}  // namespace op
}  // namespace ov
