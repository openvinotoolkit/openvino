// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_concat_with_params(ov::Shape input_shape = {1, 1, 32, 32},
                                                   ov::element::Type type = ov::element::f32);
}  // namespace utils
}  // namespace test
}  // namespace ov
