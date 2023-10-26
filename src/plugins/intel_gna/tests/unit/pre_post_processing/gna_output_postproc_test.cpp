// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "pre_post_process/converter_factory.hpp"
#include "pre_post_process/input_output_data_handler.hpp"

using namespace InferenceEngine;
using namespace pre_post_processing;

namespace testing {

typedef std::tuple<Precision,                // input precision
                   Precision,                // output precision
                   SizeVector,               // input shape
                   intel_dnn_orientation_t,  // orientation
                   float,                    // scale factor
                   bool,                     // test avx2
                   uint32_t                  // input range
                   >
    GNAOutputPrecisionParams;

template <typename U, typename T>
class GNAOutputPrecisionTest : public ::testing::TestWithParam<GNAOutputPrecisionParams> {
public:
    void SetUp() override {
        uint32_t input_range;
        std::tie(precisionIn, precisionOut, shape, orientation, sf, test_avx2, input_range) = GetParam();
        inputValues.resize(ov::shape_size(shape));
        ov::test::utils::fill_data_random(&inputValues[0], ov::shape_size(shape), input_range);

        std::transform(begin(inputValues), end(inputValues), std::back_inserter(referenceValues), [this](U i) {
            return static_cast<T>(i / sf);
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

    void compare() {
        auto total_size = ov::shape_size(shape);
        std::vector<T> pluginOutputs(total_size);
        m_input_output_handler->export_scores(&(pluginOutputs.front()),
                                              &(inputValues.front()),
                                              orientation,
                                              shape[0],
                                              shape[0],
                                              shape[1],
                                              shape[1],
                                              shape[1],
                                              precisionIn,
                                              precisionOut,
                                              sf);
        if (orientation == kDnnInterleavedOrientation) {
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; j++) {
                    float difference = std::abs(pluginOutputs[i * shape[1] + j] - referenceValues[j * shape[0] + i]);
                    EXPECT_LT(difference, std::numeric_limits<float>::epsilon());
                }
            }
        } else {
            for (int i = 0; i < total_size; ++i) {
                float difference = std::abs(pluginOutputs[i] - referenceValues[i]);
                EXPECT_LT(difference, std::numeric_limits<float>::epsilon());
            }
        }
    }

protected:
    std::shared_ptr<InputOutputDataHandler> m_input_output_handler;
    Precision precisionIn;
    Precision precisionOut;
    SizeVector shape;
    intel_dnn_orientation_t orientation;
    std::vector<T> referenceValues;
    std::vector<U> inputValues;
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

const std::vector<float> scale_factors{1.0f, 0.125f, 8.0f};

using GNAOutputPrecisionTestI8ToFp32 = GNAOutputPrecisionTest<int8_t, float>;
TEST_P(GNAOutputPrecisionTestI8ToFp32, GNAOutputPrecisionTestFp32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI8ToFp32,
                         ::testing::Combine(::testing::Values(Precision::I8),    // input precision
                                            ::testing::Values(Precision::FP32),  // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(false),            // use AVX2 version
                                            ::testing::Values(16)));             // input range

using GNAOutputPrecisionTestI16ToFp32 = GNAOutputPrecisionTest<int16_t, float>;
TEST_P(GNAOutputPrecisionTestI16ToFp32, GNAOutputPrecisionTestFp32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI16ToFp32,
                         ::testing::Combine(::testing::Values(Precision::I16),   // input precision
                                            ::testing::Values(Precision::FP32),  // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(false),            // use AVX2 version
                                            ::testing::Values(4000)));           // input range

using GNAOutputPrecisionTestI32ToFp32 = GNAOutputPrecisionTest<int32_t, float>;
TEST_P(GNAOutputPrecisionTestI32ToFp32, GNAOutputPrecisionTestFp32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI32ToFp32,
                         ::testing::Combine(::testing::Values(Precision::I32),   // input precision
                                            ::testing::Values(Precision::FP32),  // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(false),            // use AVX2 version
                                            ::testing::Values(4000)));           // input range

using GNAOutputPrecisionTestI8ToI32 = GNAOutputPrecisionTest<int8_t, int32_t>;
TEST_P(GNAOutputPrecisionTestI8ToI32, GNAOutputPrecisionTestI32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI8ToI32,
                         ::testing::Combine(::testing::Values(Precision::I8),    // input precision
                                            ::testing::Values(Precision::I32),   // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(false),            // use AVX2 version
                                            ::testing::Values(16)));             // input range

using GNAOutputPrecisionTestI16ToI32 = GNAOutputPrecisionTest<int16_t, int32_t>;
TEST_P(GNAOutputPrecisionTestI16ToI32, GNAOutputPrecisionTestI32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI16ToI32,
                         ::testing::Combine(::testing::Values(Precision::I16),   // input precision
                                            ::testing::Values(Precision::I32),   // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(false),            // use AVX2 version
                                            ::testing::Values(4000)));           // input range

using GNAOutputPrecisionTestI32ToI32 = GNAOutputPrecisionTest<int32_t, int32_t>;
TEST_P(GNAOutputPrecisionTestI32ToI32, GNAOutputPrecisionTestI32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI32ToI32,
                         ::testing::Combine(::testing::Values(Precision::I32),              // input precision
                                            ::testing::Values(Precision::I32),              // output precision
                                            ::testing::ValuesIn(input_shapes),              // input shapes
                                            ::testing::ValuesIn(orientations),              // orientations
                                            ::testing::ValuesIn(std::vector<float>{1.0f}),  // scale factors
                                            ::testing::Values(false),                       // use AVX2 version
                                            ::testing::Values(4000)));                      // input range

#ifdef HAVE_AVX2
// Below tests execute optimized functions, if AVX2 is available

using GNAOutputPrecisionTestI8ToFp32Avx = GNAOutputPrecisionTest<int8_t, float>;
TEST_P(GNAOutputPrecisionTestI8ToFp32Avx, GNAOutputPrecisionTestFp32Avx) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI8ToFp32Avx,
                         ::testing::Combine(::testing::Values(Precision::I8),    // input precision
                                            ::testing::Values(Precision::FP32),  // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(true),             // use AVX2 version
                                            ::testing::Values(16)));             // input range

using GNAOutputPrecisionTestI16ToFp32Avx = GNAOutputPrecisionTest<int16_t, float>;
TEST_P(GNAOutputPrecisionTestI16ToFp32Avx, GNAOutputPrecisionTestFp32Avx) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI16ToFp32Avx,
                         ::testing::Combine(::testing::Values(Precision::I16),   // input precision
                                            ::testing::Values(Precision::FP32),  // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(true),             // use AVX2 version
                                            ::testing::Values(4000)));           // input range

using GNAOutputPrecisionTestI32ToFp32Avx = GNAOutputPrecisionTest<int32_t, float>;
TEST_P(GNAOutputPrecisionTestI32ToFp32Avx, GNAOutputPrecisionTestFp32Avx) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(GNAOutputPrecisionTestSuite,
                         GNAOutputPrecisionTestI32ToFp32Avx,
                         ::testing::Combine(::testing::Values(Precision::I32),   // input precision
                                            ::testing::Values(Precision::FP32),  // output precision
                                            ::testing::ValuesIn(input_shapes),   // input shapes
                                            ::testing::ValuesIn(orientations),   // orientations
                                            ::testing::ValuesIn(scale_factors),  // scale factors
                                            ::testing::Values(true),             // use AVX2 version
                                            ::testing::Values(4000)));           // input range

#endif  // HAVE_AVX2
}  // namespace testing
