// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/gather_elements.hpp"

#include <vpu/private_plugin_config.hpp>

using namespace LayerTestsDefinitions;

namespace {

class GatherElementsLayerTestVPU : public GatherElementsLayerTest {
protected:
    void SetUp() override {
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        GatherElementsLayerTest::SetUp();
    }
};

TEST_P(GatherElementsLayerTestVPU, GatherElementsTests) {
    Run();
}

const std::vector<InferenceEngine::Precision> dPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> iPrecisions = {
        InferenceEngine::Precision::I32
};

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements1, GatherElementsLayerTestVPU,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2})),   // Indices shape
                            ::testing::Values(0, 1),                                  // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements2, GatherElementsLayerTestVPU,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 65, 300})),  // Data shape
                            ::testing::Values(std::vector<size_t>({2, 65, 64})),   // Indices shape
                            ::testing::Values(2),                                  // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        GatherElementsLayerTest::getTestCaseName);


}  // namespace