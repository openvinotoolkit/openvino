// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_split_conv_concat(ov::Shape input_shape = {1, 4, 20, 20},
                                                  ov::element::Type type = ov::element::f32);

std::shared_ptr<ov::Model> make_cplit_conv_concat_input_in_branch(ov::Shape input_shape = {1, 4, 20, 20},
                                                                  ov::element::Type type = ov::element::f32);

std::shared_ptr<ov::Model> make_cplit_conv_concat_nested_in_branch(ov::Shape input_shape = {1, 4, 20, 20},
                                                                   ov::element::Type type = ov::element::f32);

std::shared_ptr<ov::Model> make_cplit_conv_concat_nested_in_branch_nested_out(
    ov::Shape input_shape = {1, 4, 20, 20},
    ov::element::Type type = ov::element::f32);
}  // namespace utils
}  // namespace test
}  // namespace ov