// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "any_copy.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "gna_plugin.hpp"

using namespace InferenceEngine;

namespace testing {

class GNAPluginForInPrecisionTest : public GNAPlugin {
public:
    using GNAPlugin::GNAPlugin;
    using GNAPlugin::ImportFrames;
    void setLowPrc(bool low_precision) {
        this->gnaFlags->input_low_precision = low_precision;
    }
    void setGNADeviceHelper() {
        this->gnadevice = std::make_shared<GNADeviceHelper>();
    }
};

typedef std::tuple<InferenceEngine::Precision,   // input precision
                   InferenceEngine::SizeVector,  // input shape
                   ov::AnyMap,                   // gna config
                   bool,                         // gna device
                   bool,                         // set low precision
                   uint32_t                      // input range
                   >
    GNAInputPrecisionParams;

template <typename U, typename T>
class GNAInputPrecisionTest : public ::testing::TestWithParam<GNAInputPrecisionParams> {
public:
    void SetUp() override {
        ov::AnyMap gna_config;
        uint32_t input_range;
        std::tie(prc, shape, gna_config, is_gna_device, is_low_precision, input_range) = GetParam();
        if ((!std::is_same<T, U>::value) || ((InferenceEngine::Precision::FP32 == prc) && is_gna_device)) {
            sf = gna_config[ov::intel_gna::scale_factors_per_input.name()].as<std::map<std::string, float>>()["0"];
        }
        input_vals.resize(ov::shape_size(shape));
        CommonTestUtils::fill_data_random(&input_vals[0], ov::shape_size(shape), input_range);

        std::transform(begin(input_vals), end(input_vals), std::back_inserter(refer_vals), [this](U i) {
            return round(i * sf);
        });

        plugin.reset(new GNAPluginForInPrecisionTest(any_copy(gna_config)));
        if (is_gna_device) {
            plugin->setGNADeviceHelper();
        }
        plugin->setLowPrc(is_low_precision);
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
        plugin->ImportFrames(&(plugin_inputs.front()),
                             &(input_vals.front()),
                             prc,
                             sf,
                             orientation,
                             shape[0],
                             shape[0],
                             shape[1],
                             shape[1]);

        for (int i = 0; i < total_size; ++i) {
            EXPECT_EQ(plugin_inputs[i], refer_vals[i]);
        }
    }

protected:
    std::unique_ptr<GNAPluginForInPrecisionTest> plugin;
    InferenceEngine::Precision prc;
    InferenceEngine::SizeVector shape;
    std::vector<T> refer_vals;
    std::vector<U> input_vals;
    bool is_gna_device = false;
    bool is_low_precision = false;
    float sf = 1.0f;
    const intel_dnn_orientation_t orientation = kDnnInterleavedOrientation;
};

const std::vector<InferenceEngine::SizeVector> input_shapes{
    {1, 4},
    {1, 8},
};

using GNAInputPrecisionTestFp32toI16 = GNAInputPrecisionTest<float, int16_t>;

TEST_P(GNAInputPrecisionTestFp32toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI16,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),  // input precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.125f}}),
                            ov::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(true),   // gna device
                       ::testing::Values(false),  // use low precision
                       ::testing::Values(16)));   // input range
using GNAInputPrecisionTestFp32toI8 = GNAInputPrecisionTest<float, int8_t>;

TEST_P(GNAInputPrecisionTestFp32toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toI8,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),  // input precision
                       ::testing::ValuesIn(input_shapes),                    // input shapes
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}}),
                            ov::inference_precision(ngraph::element::i8)},
                       }),
                       ::testing::Values(true),  // gna device
                       ::testing::Values(true),  // use low precision
                       ::testing::Values(12)));  // input range

using GNAInputPrecisionTestFp32toFp32 = GNAInputPrecisionTest<float, float>;

TEST_P(GNAInputPrecisionTestFp32toFp32, GNAInputPrecisionTestFp32) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestFp32toFp32,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),  // input precision
                       ::testing::ValuesIn(input_shapes),                    // input shape
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}})},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}})},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}})},
                       }),
                       ::testing::Values(false),   // gna device
                       ::testing::Values(false),   // use low precision
                       ::testing::Values(1200)));  // input range

using GNAInputPrecisionTestI16toI16 = GNAInputPrecisionTest<int16_t, int16_t>;

TEST_P(GNAInputPrecisionTestI16toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestI16toI16,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I16),  // input precision
                       ::testing::ValuesIn(input_shapes),                   // input shapes
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                            ov::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 0.25f}}),
                            ov::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(true),   // gna device
                       ::testing::Values(false),  // use low precision
                       ::testing::Values(16)));   // input range

using GNAInputPrecisionTestI16toI8 = GNAInputPrecisionTest<int16_t, int8_t>;

TEST_P(GNAInputPrecisionTestI16toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestI16toI8,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::I16),  // input precision
                       ::testing::ValuesIn(input_shapes),                   // input shapes
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map,
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 10.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 20.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                       }),
                       ::testing::Values(true),  // gna device
                       ::testing::Values(true),  // use low precision
                       ::testing::Values(12)));  // input range

using GNAInputPrecisionTestU8toI16 = GNAInputPrecisionTest<uint8_t, int16_t>;

TEST_P(GNAInputPrecisionTestU8toI16, GNAInputPrecisionTestI16) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestU8toI16,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::U8),  // input precision
                       ::testing::ValuesIn(input_shapes),                  // input shapes
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::inference_precision(ngraph::element::i16)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 8.0f}}),
                            ov::inference_precision(ngraph::element::i16)},
                       }),
                       ::testing::Values(true),   // gna device
                       ::testing::Values(false),  // use low precision
                       ::testing::Values(16)));   // input range

using GNAInputPrecisionTestU8toI8 = GNAInputPrecisionTest<uint8_t, int8_t>;

TEST_P(GNAInputPrecisionTestU8toI8, GNAInputPrecisionTestI8) {
    compare();
}

INSTANTIATE_TEST_SUITE_P(
    GNAInputPrecisionTestSuite,
    GNAInputPrecisionTestU8toI8,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::U8),  // input precision
                       ::testing::ValuesIn(input_shapes),                  // input shapes
                       ::testing::ValuesIn(std::vector<ov::AnyMap>{
                           // gna config map
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                           {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                            ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 4.0f}}),
                            ov::inference_precision(ngraph::element::i8)},
                       }),
                       ::testing::Values(true),  // gna device
                       ::testing::Values(true),  // use low precision
                       ::testing::Values(12)));  // input range
}  // namespace testing
