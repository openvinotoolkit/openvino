// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gna/gna_config.hpp>

#include "single_layer_tests/fake_quantize.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

using ConfigType = std::map<std::string, std::string>;
const ConfigType configFP32 = {
    {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
};
const ConfigType configInt16 = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION, "I16"},
    {"GNA_SCALE_FACTOR_0", "327.67"}
};
const ConfigType configInt8 = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION, "I8"},
    {"GNA_SCALE_FACTOR_0", "327.67"}
};

/**
 * @brief specific quantisation mode to be used internally
 */
const std::vector<std::pair<std::string, ConfigType>> gnaQuantModes = {
    {"sw_fp32", configFP32},
    {"sw_exact_i16", configInt16},
    {"sw_exact_i8", configInt8},
};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1, 1, 1}, {3, 10, 5, 6}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const auto fqParams = ::testing::Combine(
    ::testing::ValuesIn(levels),
    ::testing::ValuesIn(constShapes)
);

INSTANTIATE_TEST_CASE_P(FakeQuantize, FakeQuantizeLayerTest,
    ::testing::Combine(
    fqParams,
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(inputShapes),
    ::testing::Values(CommonTestUtils::DEVICE_GNA),
    ::testing::ValuesIn(gnaQuantModes)),
    FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
