// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

using ConvertColorI420ParamsTuple = std::tuple<
        ov::Shape,                                     // Input Shape
        ov::element::Type,                             // Element type
        bool,                                          // Conversion type
        bool,                                          // 1 or 3 planes
        std::string>;                                  // Device name

class ConvertColorI420LayerTest : public testing::WithParamInterface<ConvertColorI420ParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertColorI420ParamsTuple> &obj);

protected:
    void SetUp() override;
};

//----------------------------------------

class ConvertColorI420AccuracyTest : public ConvertColorI420LayerTest {
protected:
    void GenerateInputs() override; // Generate predefined image with R/G/B combinations
    void Validate() override;       // Regular validate + percentage of acceptable deviations
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override;

    std::vector<InferenceEngine::Blob::Ptr> GetOutputs() override;
private:
    std::vector<float> expected_output;
    InferenceEngine::Blob::Ptr actual_output;
};

namespace I420TestUtils {

template <typename T>
inline void ValidateColors(const T* expected, const T* actual, size_t size, float dev_threshold, float abs_threshold = 0.01f) {
    size_t mismatches = 0;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(static_cast<float>(expected[i]) - static_cast<float>(actual[i])) > abs_threshold) {
            mismatches++;
        }
    }
    ASSERT_LT(static_cast<float>(mismatches) / size, dev_threshold) << mismatches <<
        " out of " << size << " color mismatches found which exceeds allowed threshold " << dev_threshold;
}

inline std::vector<uint8_t> color_test_image(size_t height, size_t width, int b_step) {
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
                size_t u_index = b_offset + height * width + y * width / 2 + x;
                size_t v_index = u_index + height * width / 4;
                input_yuv[u_index] = u_val;
                input_yuv[v_index] = v_val;
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

} // namespace I420TestUtils
} // namespace LayerTestsDefinitions
