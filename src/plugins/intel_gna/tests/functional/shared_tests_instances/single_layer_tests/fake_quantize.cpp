// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/fake_quantize.hpp"

#include <gna/gna_config.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

using ConfigType = std::map<std::string, std::string>;
const ConfigType configFP32 = {
    {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
};
const ConfigType configInt16 = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                                {InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION, "I16"},
                                {"GNA_SCALE_FACTOR_0", "327.67"}};
const ConfigType configInt8 = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                               {InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION, "I8"},
                               {"GNA_SCALE_FACTOR_0", "327.67"}};

/**
 * @brief specific quantisation mode to be used internally
 */
const std::vector<std::pair<std::string, ConfigType>> gnaQuantModes = {
    {"sw_fp32", configFP32},
    // TODO: support FakeQuantize in integer mode
    //    {"sw_exact_i16", configInt16},
    //    {"sw_exact_i8", configInt8},
};

const std::vector<std::vector<size_t>> inputShapes =
    {{3, 10, 5, 6}, {1, 1, 1, 1}, {1, 8, 8, 256}, {1, 2, 2, 2}, {1, 3, 4, 5}, {8}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256, UINT32_MAX};

const std::vector<std::vector<float>> fqArgs = {{}};
const std::vector<std::vector<float>> inputParams = {{-10, 10, 0.1}, {}};

const std::vector<float> fqInputMin = {0, 3};
const std::vector<float> fqInputMax = {10, 7};
const std::vector<float> fqOutputMin = {1, 3};
const std::vector<float> fqOutputMax = {7, 6};

std::vector<std::vector<float>> getInputOutputShapes(const std::vector<float> inputsMin,
                                                     const std::vector<float> inputsMax,
                                                     const std::vector<float> OutputsMin,
                                                     const std::vector<float> OutputsMax,
                                                     std::vector<std::vector<float>> fqArg) {
    for (const auto& inputMin : inputsMin) {
        for (const auto& inputMax : inputsMax) {
            for (const auto& outputMin : OutputsMin) {
                for (const auto& outputMax : OutputsMax) {
                    fqArg.push_back({inputMin, inputMax, outputMin, outputMax});
                }
            }
        }
    }
    return fqArg;
}

const auto fqParams = ::testing::Combine(::testing::ValuesIn(levels),
                                         ::testing::ValuesIn(constShapes),
                                         ::testing::ValuesIn(fqArgs),
                                         ::testing::ValuesIn(inputParams),
                                         ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize,
                         FakeQuantizeLayerTest,
                         ::testing::Combine(fqParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(gnaQuantModes)),
                         FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
