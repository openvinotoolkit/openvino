// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/tensor.hpp>

namespace ov {
namespace test {
ov::runtime::Tensor create_and_fill_tensor(
    const ov::element::Type element_type,
    const ov::Shape& shape,
    const uint32_t range = 10,
    const int32_t start_from = 0,
    const int32_t resolution = 1,
    const int seed = 1);
}  // namespace test
}  // namespace ov