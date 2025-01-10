// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_single_conv(ov::Shape input_shape = {1, 3, 24, 24},
                                            ov::element::Type type = ov::element::f32);
}  // namespace utils
}  // namespace test
}  // namespace ov
