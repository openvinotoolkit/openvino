// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "pre_post_process/converter_factory.hpp"
#include "pre_post_process/input_output_data_handler.hpp"

using namespace InferenceEngine;
using namespace pre_post_processing;

namespace testing {

typedef std::tuple<Precision,                // input precision
                   SizeVector,               // input shape
                   intel_dnn_orientation_t,  // orientation
                   float,                    // scale factor
                   bool,                     // gna device
                   bool,                     // set low precision
                   bool,                     // test avx2
                   uint32_t                  // input range
                   >
    GNAInputPrecisionParams;

template <typename U, typename T>
class GNAInputPrecisionTest : public ::testing::TestWithParam<GNAInputPrecisionParams> {
public:
    void SetUp() override {
        uint32_t input_range;
        std::tie(prc, shape, orientation, sf, is_gna_device, is_low_precision, test_avx2, input_range) = GetParam();
        input_vals.resize(ov::shape_size(shape));
        ov::test::utils::fill_data_random(&input_vals[0], ov::shape_size(shape), input_range);

        std::transform(begin(input_vals), end(input_vals), std::back_inserter(refer_vals), [this](U i) {
            return round(i * sf);
        });

        if (test_avx2) {
            auto converter = ConverterFactory::create_converter();
            if (converter == nullptr) {
                GTEST_SKIP() << "Tests compiled with with AVX2 support, but AVX2 unavailable at runtime";
            }
            m_input_output_handler = std::make_shared<InputOutputDataHandler>(converter);
        } else {
            m_input_output_handler = std::make_shared<InputOutputDataHandler>(nullptr);
        }
    }

    T round(float src) {
        float rounding_value = is_gna_device ? ((src > 0) ? 0.5f : -0.5f) : 0.0f;
        float value = src + rounding_value;
        if (value > std::numeric_limits<T>::max()) {
            return std::numeric_limits<T>::max();
        } else if (value < std::numeric_limits<T>::min()) {
            return std::numeric_limits<T>::min();
        }
        return static_cast<T>(value);
    }

    void compare() {
        auto total_size = ov::shape_size(shape);
        std::vector<T> plugin_inputs(total_size);
        m_input_output_handler->import_frames(&(plugin_inputs.front()),
                                              &(input_vals.front()),
                                              prc,
                                              sf,
                                              orientation,
                                              shape[0],
                                              shape[0],
                                              shape[1],
                                              shape[1],
                                              is_low_precision,
                                              is_gna_device);
        if (orientation == kDnnInterleavedOrientation) {
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; j++) {
                    float difference = std::abs(plugin_inputs[j * shape[0] + i] - refer_vals[i * shape[1] + j]);
                    EXPECT_LT(difference, std::numeric_limits<float>::epsilon());
                }
            }
        } else {
            for (int i = 0; i < total_size; ++i) {
                float difference = std::abs(plugin_inputs[i] - refer_vals[i]);
                EXPECT_LT(difference, std::numeric_limits<float>::epsilon());
            }
        }
    }

protected:
    std::shared_ptr<InputOutputDataHandler> m_input_output_handler;
    Precision prc;
    SizeVector shape;
    intel_dnn_orientation_t orientation;
    std::vector<T> refer_vals;
    std::vector<U> input_vals;
    bool is_gna_device = false;
    bool is_low_precision = false;
    bool test_avx2 = false;
    float sf = 1.0f;
};

const std::vector<SizeVector> input_shapes{
    {1, 20},
    {4, 8},
    {31, 1},
    {9, 3},
};

const std::vector<intel_dnn_orientation_t> orientations{kDnnInterleavedOrientation, kDnnNonInterleavedOrientation};

using GNAInputPrecisionTestFp32toI16 = GNAInputPrecisionTest<float, int16_t>;

TEST_P(GNAInputPrecisionTestFp32toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI16,
    ::testing::Combine(::testing::Values(Precision::FP32),                           // input precision
                       ::testing::ValuesIn(input_shapes),                            // input shapes
                       ::testing::ValuesIn(orientations),                            // orientations
                       ::testing::ValuesIn(std::vector<float>{1.0f, 8.0f, 0.125f}),  // scale factors
                       ::testing::Values(true),                                      // gna device
                       ::testing::Values(false),                                     // use low precision
                       ::testing::Values(false),                                     // use AVX2 version
                       ::testing::Values(16)));                                      // input range
using GNAInputPrecisionTestFp32toI8 = GNAInputPrecisionTest<float, int8_t>;

TEST_P(GNAInputPrecisionTestFp32toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI8,
    ::testing::Combine(::testing::Values(Precision::FP32),                          // input precision
                       ::testing::ValuesIn(input_shapes),                           // input shapes
                       ::testing::ValuesIn(orientations),                           // orientations
                       ::testing::ValuesIn(std::vector<float>{1.0f, 4.0f, 0.25f}),  // scale factors
                       ::testing::Values(true),                                     // gna device
                       ::testing::Values(true),                                     // use low precision
                       ::testing::Values(false),                                    // use AVX2 version
                       ::testing::Values(12)));                                     // input range

using GNAInputPrecisionTestFp32toFp32 = GNAInputPrecisionTest<float, float>;

TEST_P(GNAInputPrecisionTestFp32toFp32, GNAInputPrecisionTestFp32) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite,
                         GNAInputPrecisionTestFp32toFp32,
                         ::testing::Combine(::testing::Values(Precision::FP32),             // input precision
                                            ::testing::ValuesIn(input_shapes),              // input shape
                                            ::testing::ValuesIn(orientations),              // orientations
                                            ::testing::ValuesIn(std::vector<float>{1.0f}),  // scale factors
                                            ::testing::Values(false),                       // gna device
                                            ::testing::Values(false),                       // use low precision
                                            ::testing::Values(false),                       // use AVX2 version
                                            ::testing::Values(1200)));                      // input range

using GNAInputPrecisionTestI16toI16 = GNAInputPrecisionTest<int16_t, int16_t>;

TEST_P(GNAInputPrecisionTestI16toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite,
                         GNAInputPrecisionTestI16toI16,
                         ::testing::Combine(::testing::Values(Precision::I16),              // input precision
                                            ::testing::ValuesIn(input_shapes),              // input shapes
                                            ::testing::ValuesIn(orientations),              // orientations
                                            ::testing::ValuesIn(std::vector<float>{1.0f}),  // scale factors
                                            ::testing::Values(true),                        // gna device
                                            ::testing::Values(false),                       // use low precision
                                            ::testing::Values(false),                       // use AVX2 version
                                            ::testing::Values(16)));                        // input range

using GNAInputPrecisionTestI16toI8 = GNAInputPrecisionTest<int16_t, int8_t>;

TEST_P(GNAInputPrecisionTestI16toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestI16toI8,
    ::testing::Combine(::testing::Values(Precision::I16),                            // input precision
                       ::testing::ValuesIn(input_shapes),                            // input shapes
                       ::testing::ValuesIn(orientations),                            // orientations
                       ::testing::ValuesIn(std::vector<float>{1.0f, 10.0f, 20.0f}),  // scale factors
                       ::testing::Values(true),                                      // gna device
                       ::testing::Values(true),                                      // use low precision
                       ::testing::Values(false),                                     // use AVX2 version
                       ::testing::Values(12)));                                      // input range

using GNAInputPrecisionTestU8toI16 = GNAInputPrecisionTest<uint8_t, int16_t>;

TEST_P(GNAInputPrecisionTestU8toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite,
                         GNAInputPrecisionTestU8toI16,
                         ::testing::Combine(::testing::Values(Precision::U8),                     // input precision
                                            ::testing::ValuesIn(input_shapes),                    // input shapes
                                            ::testing::ValuesIn(orientations),                    // orientations
                                            ::testing::ValuesIn(std::vector<float>{1.0f, 8.0f}),  // scale factors
                                            ::testing::Values(true),                              // gna device
                                            ::testing::Values(false),                             // use low precision
                                            ::testing::Values(false),                             // use AVX2 version
                                            ::testing::Values(16)));                              // input range

using GNAInputPrecisionTestU8toI8 = GNAInputPrecisionTest<uint8_t, int8_t>;

TEST_P(GNAInputPrecisionTestU8toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite,
                         GNAInputPrecisionTestU8toI8,
                         ::testing::Combine(::testing::Values(Precision::U8),               // input precision
                                            ::testing::ValuesIn(input_shapes),              // input shapes
                                            ::testing::ValuesIn(orientations),              // orientations
                                            ::testing::ValuesIn(std::vector<float>{1.0f}),  // scale factors
                                            ::testing::Values(true),                        // gna device
                                            ::testing::Values(true),                        // use low precision
                                            ::testing::Values(false),                       // use AVX2 version
                                            ::testing::Values(12)));                        // input range

#ifdef HAVE_AVX2
// Below tests execute optimized functions, if AVX2 is available

using GNAInputPrecisionTestFp32toI16Avx = GNAInputPrecisionTest<float, int16_t>;

TEST_P(GNAInputPrecisionTestFp32toI16Avx, GNAInputPrecisionTestI16Avx) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI16Avx,
    ::testing::Combine(::testing::Values(Precision::FP32),                           // input precision
                       ::testing::ValuesIn(input_shapes),                            // input shapes
                       ::testing::ValuesIn(orientations),                            // orientations
                       ::testing::ValuesIn(std::vector<float>{1.0f, 8.0f, 0.125f}),  // scale factors
                       ::testing::Values(true),                                      // gna device
                       ::testing::Values(false),                                     // use low precision
                       ::testing::Values(true),                                      // use AVX2 version
                       ::testing::Values(16)));                                      // input range

using GNAInputPrecisionTestFp32toI8Avx = GNAInputPrecisionTest<float, int8_t>;

TEST_P(GNAInputPrecisionTestFp32toI8Avx, GNAInputPrecisionTestI8Avx) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI8Avx,
    ::testing::Combine(::testing::Values(Precision::FP32),                          // input precision
                       ::testing::ValuesIn(input_shapes),                           // input shapes
                       ::testing::ValuesIn(orientations),                           // orientations
                       ::testing::ValuesIn(std::vector<float>{1.0f, 4.0f, 0.25f}),  // scale factors
                       ::testing::Values(true),                                     // gna device
                       ::testing::Values(true),                                     // use low precision
                       ::testing::Values(true),                                     // use AVX2 version
                       ::testing::Values(12)));                                     // input range
#endif                                                                              // HAVE_AVX2
}  // namespace testing
