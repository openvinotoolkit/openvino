// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/tensor.hpp>

namespace ov {
namespace test {
namespace utils {
ov::runtime::Tensor create_and_fill_tensor(
        const ov::element::Type element_type,
        const ov::Shape &shape,
        const uint32_t range = 10,
        const int32_t start_from = 0,
        const int32_t resolution = 1,
        const int seed = 1);

ov::runtime::Tensor create_and_fill_tensor_unique_sequence(
        const ov::element::Type element_type,
        const ov::Shape& shape,
        const int32_t start_from = 0,
        const int32_t resolution = 1,
        const int seed = 1);

void compare(
        const ov::runtime::Tensor &expected,
        const ov::runtime::Tensor &actual,
        const double abs_threshold,
        const double rel_threshold);
}  // namespace utils
}  // namespace test
}  // namespace ov