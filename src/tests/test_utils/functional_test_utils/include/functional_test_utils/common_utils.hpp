// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/core/preprocess/color_format.hpp"

namespace ov {
namespace test {
namespace utils {

std::vector<uint8_t> color_test_image(size_t height, size_t width, int b_step, ov::preprocess::ColorFormat format);
}  // namespace utils
}  // namespace test
}  // namespace ov
