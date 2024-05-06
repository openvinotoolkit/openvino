// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/common_utils.hpp"

namespace ov {
namespace test {
namespace utils {
std::vector<uint8_t> color_test_image(size_t height, size_t width, int b_step, ov::preprocess::ColorFormat format) {
    // Test all possible r/g/b values within dimensions
    int b_dim = 255 / b_step + 1;
    auto input_yuv = std::vector<uint8_t>(height * b_dim * width * 3 / 2);
    for (int b = 0; b <= 255; b += b_step) {
        for (size_t y = 0; y < height / 2; y++) {
            for (size_t x = 0; x < width / 2; x++) {
                int r = static_cast<int>(y) * 512 / static_cast<int>(height);
                int g = static_cast<int>(x) * 512 / static_cast<int>(width);
                // Can't use random y/u/v for testing as this can lead to invalid R/G/B values
                int y_val = ((66 * r + 129 * g + 25 * b + 128) / 256) + 16;
                int u_val = ((-38 * r - 74 * g + 112 * b + 128) / 256) + 128;
                int v_val = ((112 * r - 94 * g + 18 * b + 128) / 256) + 128;

                size_t b_offset = height * width * b / b_step * 3 / 2;
                if (ov::preprocess::ColorFormat::I420_SINGLE_PLANE == format ||
                    ov::preprocess::ColorFormat::I420_THREE_PLANES == format) {
                    size_t u_index = b_offset + height * width + y * width / 2 + x;
                    size_t v_index = u_index + height * width / 4;
                    input_yuv[u_index] = u_val;
                    input_yuv[v_index] = v_val;
                } else {
                    size_t uv_index = b_offset + height * width + y * width + x * 2;
                    input_yuv[uv_index] = u_val;
                    input_yuv[uv_index + 1] = v_val;
                }
                size_t y_index = b_offset + y * 2 * width + x * 2;
                input_yuv[y_index] = y_val;
                input_yuv[y_index + 1] = y_val;
                input_yuv[y_index + width] = y_val;
                input_yuv[y_index + width + 1] = y_val;
            }
        }
    }
    return input_yuv;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
