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
#include "gna_plugin.hpp"
#include "preprocessing.hpp"

using namespace InferenceEngine;

namespace testing {

class GNAPluginForInPrecisionTest : public GNAPluginNS::GNAPlugin {
public:
using GNAPlugin::GNAPlugin;
    void setLowPrc(bool low_precision) {
        this->gnaFlags->input_low_precision = low_precision;
    }

    bool isLowPrc() const {
        return this->gnaFlags->input_low_precision;
    }

    void setGNADeviceHelper() {
        this->gnadevice = std::make_shared<GNADeviceHelper>();
    }

    bool isGnaDevicePresent() const {
        return gnadevice.get() != nullptr;
    }

    void setAvx2Support(bool testAvx2) {
        this->isAvx2Supported = testAvx2;
    }

    bool isAvx2Support() const {
        return isAvx2Supported;
    }
};

typedef std::tuple<InferenceEngine::Precision, // input precision
        InferenceEngine::SizeVector,           // input shape
        intel_dnn_orientation_t,               // orientation
        ov::AnyMap,                            // gna config
        bool,                                  // gna device
        bool,                                  // set low precision
        bool,                                  // test avx2
        uint32_t                               // input range
> GNAInputPrecisionParams;

template<typename U, typename T>
class GNAInputPrecisionTest: public ::testing::TestWithParam<GNAInputPrecisionParams> {
public:
    void SetUp() override {
        ov::AnyMap gna_config;
        uint32_t input_range;
        std::tie(prc, shape, orientation, gna_config, is_gna_device, is_low_precision, testAvx2, input_range) =
            GetParam();
        if ((!std::is_same<T, U>::value) || ((InferenceEngine::Precision::FP32 == prc) && is_gna_device)) {
            sf = gna_config[ov::intel_gna::scale_factors_per_input.name()].as<std::map<std::string, float>>()["0"];
        }
        input_vals.resize(ov::shape_size(shape));
        CommonTestUtils::fill_data_random(&input_vals[0], ov::shape_size(shape), input_range);

        std::transform(begin(input_vals), end(input_vals),
            std::back_inserter(refer_vals), [this](U i) {return round(i*sf); });

        plugin.reset(new GNAPluginForInPrecisionTest(any_copy(gna_config)));
        if (is_gna_device) {
            plugin->setGNADeviceHelper();
        }
        plugin->setLowPrc(is_low_precision);
        plugin->setAvx2Support(testAvx2);
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
        GNAPluginNS::ImportFrames(&(plugin_inputs.front()),
                                  &(input_vals.front()),
                                  prc,
                                  sf,
                                  orientation,
                                  shape[0],
                                  shape[0],
                                  shape[1],
                                  shape[1],
                                  plugin->isLowPrc(),
                                  plugin->isGnaDevicePresent(),
                                  plugin->isAvx2Support());
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
    std::unique_ptr<GNAPluginForInPrecisionTest> plugin;
    InferenceEngine::Precision prc;
    InferenceEngine::SizeVector shape;
    intel_dnn_orientation_t orientation;
    std::vector<T> refer_vals;
    std::vector<U> input_vals;
    bool is_gna_device = false;
    bool is_low_precision = false;
    bool testAvx2 = false;
    float sf = 1.0f;
};

const std::vector<InferenceEngine::SizeVector> input_shapes{
    {1, 20},
    {41, 81},
};

const std::vector<intel_dnn_orientation_t> orientations{
    kDnnInterleavedOrientation,
    kDnnNonInterleavedOrientation
};

using GNAInputPrecisionTestFp32toI16 = GNAInputPrecisionTest<float, int16_t>;

TEST_P(GNAInputPrecisionTestFp32toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestFp32toI16,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::FP32),  // input precision
                            ::testing::ValuesIn(input_shapes),                    // input shapes
                            ::testing::ValuesIn(orientations),                    // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {         // gna config map
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
                            ::testing::Values(true),                              // gna device
                            ::testing::Values(false),                             // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(16)));                              // input range
using GNAInputPrecisionTestFp32toI8 = GNAInputPrecisionTest<float, int8_t>;

TEST_P(GNAInputPrecisionTestFp32toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestFp32toI8,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::FP32), // input precision
                            ::testing::ValuesIn(input_shapes),                   // input shapes
                            ::testing::ValuesIn(orientations),                    // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {        // gna config map
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                            }),
                            ::testing::Values(true),                              // gna device
                            ::testing::Values(true),                              // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(12)));                              // input range

using GNAInputPrecisionTestFp32toFp32 = GNAInputPrecisionTest<float, float>;

TEST_P(GNAInputPrecisionTestFp32toFp32, GNAInputPrecisionTestFp32) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestFp32toFp32,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::FP32),   // input precision
                            ::testing::ValuesIn(input_shapes),                     // input shape
                            ::testing::ValuesIn(orientations),                     // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {          // gna config map
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}})},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}})},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}})},
                            }),
                            ::testing::Values(false),                             // gna device
                            ::testing::Values(false),                             // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(1200)));                            // input range

using GNAInputPrecisionTestI16toI16 = GNAInputPrecisionTest<int16_t, int16_t>;

TEST_P(GNAInputPrecisionTestI16toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestI16toI16,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::I16),   // input precision
                            ::testing::ValuesIn(input_shapes),                    // input shapes
                            ::testing::ValuesIn(orientations),                    // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {         // gna config map
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i16)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i16)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}}),
                                    ov::hint::inference_precision(ngraph::element::i16)},
                            }),
                            ::testing::Values(true),                              // gna device
                            ::testing::Values(false),                             // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(16)));                              // input range

using GNAInputPrecisionTestI16toI8 = GNAInputPrecisionTest<int16_t, int8_t>;

TEST_P(GNAInputPrecisionTestI16toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestI16toI8,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::I16),    // input precision
                            ::testing::ValuesIn(input_shapes),                     // input shapes
                            ::testing::ValuesIn(orientations),                     // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {          // gna config map,
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 10.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 20.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                            }),
                            ::testing::Values(true),                              // gna device
                            ::testing::Values(true),                              // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(12)));                              // input range

using GNAInputPrecisionTestU8toI16 = GNAInputPrecisionTest<uint8_t, int16_t>;

TEST_P(GNAInputPrecisionTestU8toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestU8toI16,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::U8),     // input precision
                            ::testing::ValuesIn(input_shapes),                     // input shapes
                            ::testing::ValuesIn(orientations),                     // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {          // gna config map
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i16)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i16)},
                            }),
                            ::testing::Values(true),                              // gna device
                            ::testing::Values(false),                             // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(16)));                              // input range

using GNAInputPrecisionTestU8toI8 = GNAInputPrecisionTest<uint8_t, int8_t>;

TEST_P(GNAInputPrecisionTestU8toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(GNAInputPrecisionTestSuite, GNAInputPrecisionTestU8toI8,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::U8),    // input precision
                            ::testing::ValuesIn(input_shapes),                    // input shapes
                            ::testing::ValuesIn(orientations),                    // orientations
                            ::testing::ValuesIn(std::vector<ov::AnyMap> {         // gna config map
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                                {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                                    ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                                    ov::hint::inference_precision(ngraph::element::i8)},
                            }),
                            ::testing::Values(true),                              // gna device
                            ::testing::Values(true),                              // use low precision
                            ::testing::Values(false),                             // use AVX2 version
                            ::testing::Values(12)));                              // input range

using GNAInputPrecisionTestFp32toI16Avx = GNAInputPrecisionTest<float, int16_t>;

TEST_P(GNAInputPrecisionTestFp32toI16Avx, GNAInputPrecisionTestI16Avx) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI16Avx,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),  // input precision
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
                       ::testing::Values(true),                              // gna device
                       ::testing::Values(false),                             // use low precision
                       ::testing::Values(ov::intel_gna::isAvx2Supported()),  // use AVX2 version
                       ::testing::Values(16)));                              // input range

using GNAInputPrecisionTestFp32toI8Avx = GNAInputPrecisionTest<float, int8_t>;

TEST_P(GNAInputPrecisionTestFp32toI8Avx, GNAInputPrecisionTestI8Avx) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI8Avx,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),  // input precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(orientations),                    // orientations
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::hint::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                            ov::hint::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}}),
                            ov::hint::inference_precision(ngraph::element::i8)},
                       }),
                       ::testing::Values(true),                              // gna device
                       ::testing::Values(true),                              // use low precision
                       ::testing::Values(ov::intel_gna::isAvx2Supported()),  // use AVX2 version
                       ::testing::Values(12)));                              // input range
} // namespace testing
