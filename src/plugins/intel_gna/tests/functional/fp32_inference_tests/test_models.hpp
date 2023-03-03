// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/openvino.hpp"

std::shared_ptr<ov::Model> eltwise_add_model();
std::shared_ptr<ov::Model> fc_with_padding_after_split_model();
std::shared_ptr<ov::Model> slice_model_with_aligned_outputs();
std::shared_ptr<ov::Model> two_fc_with_padding_after_slice_model();
std::shared_ptr<ov::Model> scaleshift_3d_model();
std::shared_ptr<ov::Model> input_split_concat_model();
std::shared_ptr<ov::Model> input_split_concat_unaligned_model();
std::shared_ptr<ov::Model> power_with_scale_factor_model();
std::shared_ptr<ov::Model> lstm_cell_only_model();
std::shared_ptr<ov::Model> lstm_cell_only_model_unaligned();
std::shared_ptr<ov::Model> two_inputs_to_affine_model();
std::shared_ptr<ov::Model> reshape_convolution_less_than_48_filters();
std::shared_ptr<ov::Model> two_outputs_model();
std::shared_ptr<ov::Model> two_outputs_relu_model();
