// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <vector>

#include "any_copy.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "gna_data_types.hpp"
#include "gna_plugin.hpp"
#include "preprocessing.hpp"

using namespace InferenceEngine;

namespace testing {

class GNAPluginForOutPrecisionTest : public GNAPluginNS::GNAPlugin {
public:
using GNAPlugin::GNAPlugin;
    void setGNADeviceHelper() {
        gnadevice = std::make_shared<GNADeviceHelper>();
    }

    bool isGnaDevicePresent() const {
        return gnadevice.get() != nullptr;
    }

    void setAvx2Support(bool testAvx2) {
        isAvx2Supported = testAvx2;
    }

    bool isAvx2Support() const {
        return isAvx2Supported;
    }
};

typedef std::tuple<InferenceEngine::Precision,   // input precision
                   InferenceEngine::Precision,   // output precision
                   InferenceEngine::SizeVector,  // input shape
                   intel_dnn_orientation_t,      // orientation
                   ov::AnyMap,                   // gna config
                   bool,                         // test avx2
                   uint32_t                      // input range
                   >
    GNAOutputPrecisionParams;

template<typename U, typename T>
class GNAOutputPrecisionTest : public ::testing::TestWithParam<GNAOutputPrecisionParams> {
public:
    void SetUp() override {
        ov::AnyMap gna_config;
        uint32_t input_range;
        std::tie(precisionIn, precisionOut, shape, orientation, gna_config, testAvx2, input_range) =
            GetParam();
        if (!std::is_same<T, U>::value) {
            sf = gna_config[ov::intel_gna::scale_factors_per_input.name()].as<std::map<std::string, float>>()["0"];
        }
        inputValues.resize(ov::shape_size(shape));
        CommonTestUtils::fill_data_random(&inputValues[0], ov::shape_size(shape), input_range);

        std::transform(begin(inputValues), end(inputValues), std::back_inserter(referenceValues), [this](U i) {
            return static_cast<T>(i / sf);
        });

        plugin.reset(new GNAPluginForOutPrecisionTest(any_copy(gna_config)));
        plugin->setAvx2Support(testAvx2);
    }

    void compare() {
        auto total_size = ov::shape_size(shape);
        std::vector<T> pluginOutputs(total_size);
        GNAPluginNS::ExportScores(&(pluginOutputs.front()),
                                  &(inputValues.front()),
                                  orientation,
                                  shape[0],
                                  shape[0],
                                  shape[1],
                                  shape[1],
                                  shape[1],
                                  precisionIn,
                                  precisionOut,
                                  sf,
                                  plugin->isAvx2Support());
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
    std::unique_ptr<GNAPluginForOutPrecisionTest> plugin;
    InferenceEngine::Precision precisionIn;
    InferenceEngine::Precision precisionOut;
    InferenceEngine::SizeVector shape;
    intel_dnn_orientation_t orientation;
    std::vector<T> referenceValues;
    std::vector<U> inputValues;
    bool testAvx2 = false;
    float sf = 1.0f;
};

const std::vector<InferenceEngine::SizeVector> input_shapes{
    {1, 20},
    {4, 8},
};

const std::vector<intel_dnn_orientation_t> orientations{
    kDnnInterleavedOrientation,
    kDnnNonInterleavedOrientation
};



using GNAOutputPrecisionTestI8ToFp32 = GNAOutputPrecisionTest<int8_t, float>;
TEST_P(GNAOutputPrecisionTestI8ToFp32, GNAOutputPrecisionTestFp32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI8ToFp32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I8),    // input precision
                       ::testing::Values(InferenceEngine::Precision::FP32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(false), // use AVX2 version
                       ::testing::Values(16)));  // input range



using GNAOutputPrecisionTestI16ToFp32 = GNAOutputPrecisionTest<int16_t, float>;
TEST_P(GNAOutputPrecisionTestI16ToFp32, GNAOutputPrecisionTestFp32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI16ToFp32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I16),    // input precision
                       ::testing::Values(InferenceEngine::Precision::FP32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(false),   // use AVX2 version
                       ::testing::Values(4000)));  // input range



using GNAOutputPrecisionTestI32ToFp32 = GNAOutputPrecisionTest<int32_t, float>;
TEST_P(GNAOutputPrecisionTestI32ToFp32, GNAOutputPrecisionTestFp32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI32ToFp32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I32),   // input precision
                       ::testing::Values(InferenceEngine::Precision::FP32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(false),   // use AVX2 version
                       ::testing::Values(4000)));  // input range

using GNAOutputPrecisionTestI8ToI32 = GNAOutputPrecisionTest<int8_t, int32_t>;
TEST_P(GNAOutputPrecisionTestI8ToI32, GNAOutputPrecisionTestI32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI8ToI32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I8),    // input precision
                       ::testing::Values(InferenceEngine::Precision::I32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(false),  // use AVX2 version
                       ::testing::Values(16)));  // input range

using GNAOutputPrecisionTestI16ToI32 = GNAOutputPrecisionTest<int16_t, int32_t>;
TEST_P(GNAOutputPrecisionTestI16ToI32, GNAOutputPrecisionTestI32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI16ToI32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I16),   // input precision
                       ::testing::Values(InferenceEngine::Precision::I32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(false),   // use AVX2 version
                       ::testing::Values(4000)));  // input range

using GNAOutputPrecisionTestI32ToI32 = GNAOutputPrecisionTest<int32_t, int32_t>;
TEST_P(GNAOutputPrecisionTestI32ToI32, GNAOutputPrecisionTestI32) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI32ToI32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I32),   // input precision
                       ::testing::Values(InferenceEngine::Precision::I32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(false),   // use AVX2 version
                       ::testing::Values(4000)));  // input range


#ifdef HAVE_AVX2

using GNAOutputPrecisionTestI8ToFp32Avx = GNAOutputPrecisionTest<int8_t, float>;
TEST_P(GNAOutputPrecisionTestI8ToFp32Avx, GNAOutputPrecisionTestFp32Avx) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI8ToFp32Avx,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I8),    // input precision
                       ::testing::Values(InferenceEngine::Precision::FP32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(ov::intel_gna::isAvx2Supported()),  // use AVX2 version
                       ::testing::Values(16)));                              // input range

using GNAOutputPrecisionTestI16ToFp32Avx = GNAOutputPrecisionTest<int16_t, float>;
TEST_P(GNAOutputPrecisionTestI16ToFp32Avx, GNAOutputPrecisionTestFp32Avx) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI16ToFp32Avx,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I16),   // input precision
                       ::testing::Values(InferenceEngine::Precision::FP32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(ov::intel_gna::isAvx2Supported()),  // use AVX2 version
                       ::testing::Values(4000)));                            // input range

using GNAOutputPrecisionTestI32ToFp32Avx = GNAOutputPrecisionTest<int32_t, float>;
TEST_P(GNAOutputPrecisionTestI32ToFp32Avx, GNAOutputPrecisionTestFp32Avx) {
    compare();
}
INSTANTIATE_TEST_SUITE_P(
    GNAOutputPrecisionTestSuite,
    GNAOutputPrecisionTestI32ToFp32Avx,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I32),   // input precision
                       ::testing::Values(InferenceEngine::Precision::FP32),  // output precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::hint::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(ov::intel_gna::isAvx2Supported()),  // use AVX2 version
                       ::testing::Values(4000)));                            // input range
#endif //HAVE_AVX2
}  // namespace testing
