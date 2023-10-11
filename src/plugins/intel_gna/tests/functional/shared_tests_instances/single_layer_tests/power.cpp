// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/power.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

class PwlUniformDesignPowerLayerTest : public PowerLayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PowerParamsTuple>& obj) {
        std::ostringstream results;
        results << PowerLayerTest::getTestCaseName(obj);
        results << "_configItem=" << config.first << "_" << config.second;
        return results.str();
    }

protected:
    void SetUp() override {
        PowerLayerTest::SetUp();
        configuration.emplace(config);
    }

private:
    static const std::pair<std::string, std::string> config;
};

const std::pair<std::string, std::string> PwlUniformDesignPowerLayerTest::config = {"GNA_PWL_UNIFORM_DESIGN", "YES"};

std::vector<std::vector<std::vector<size_t>>> inShapes =
    {{{1, 8}}, {{2, 16}}, {{3, 32}}, {{4, 64}}, {{5, 128}}, {{6, 256}}, {{7, 512}}, {{8, 1024}}, {{5}}, {{8}}};

std::vector<std::vector<float>> Power = {
    {0.0f},
    {0.5f},
    {1.0f},
    {1.1f},
    {1.5f},
    {2.0f},
};

std::vector<std::vector<float>> PowerPwlUniformDesign = {
    {0.0f},
    {0.5f},
    {1.0f},
    {1.1f},
    {1.5f},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_power,
                         PowerLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(Power)),
                         PowerLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_power,
                         PwlUniformDesignPowerLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(PowerPwlUniformDesign)),
                         PwlUniformDesignPowerLayerTest::getTestCaseName);

TEST_P(PwlUniformDesignPowerLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace
