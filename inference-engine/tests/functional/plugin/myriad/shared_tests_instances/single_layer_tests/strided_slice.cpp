// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/myriad_common_test_utils.hpp"

#include "vpu/private_plugin_config.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

typedef std::map<std::string, std::string> Config;

std::vector<StridedSliceSpecificParams> testCases = {
    { { 1, 12, 100 }, { 0, 9, 0 }, { 0, 11, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
    { { 1, 12, 100 }, { 0, 9 }, { 0, 11 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
    { { 1, 12, 100 }, { 0, 9 }, { 0, 11 }, { 1, 1 }, { 1, 0 }, { 1, 0 },  { 0, 0 },  { 0, 0 },  { 0, 0 } },
    { { 1, 12, 100 }, { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
    { { 1, 12, 100 }, { 0, 8, 0 }, { 0, 9, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  {},  {},  {} },
    { { 1, 12, 100 }, { 0, 4, 0 }, { 0, 9, 0 }, { 1, 2, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
    { { 1, 12, 100 }, { 0, 0, 0 }, { 0, 11, 0 }, { 1, 2, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
    { { 1, 12, 100 }, { 0, -8, 0 }, { 0, -6, 0 }, { 1, 2, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  {},  {},  {} },
    { { 1, 2, 2, 2 }, { 0, 0, 0, 0 }, { 1, 2, 2, 2 }, { 1, 1, 1, 1 }, {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {} },
    { { 1, 2, 2, 2 }, { 0, 1, 1, 1 }, { 1, 2, 2, 2 }, { 1, 1, 1, 1 }, {0, 0, 0, 0}, {1, 1, 1, 1},  {},  {},  {} },
    { { 1, 2, 2, 2 }, { 0, 1, 1, 1 }, { 1, 2, 2, 2 }, { 1, 1, 1, 1 }, {0, 0, 0, 0}, {0, 0, 0, 0},  {},  {},  {} },
    { { 1, 2, 4, 3 }, { 0, 0, 0, 0 }, { 1, 2, 4, 3 }, { 1, 1, 2, 1 }, {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {} },
    { { 1, 2, 4, 2 }, { 0, 0, 0, 1 }, { 1, 2, 4, 2 }, { 1, 1, 2, 1 }, {0, 1, 1, 0}, {1, 1, 0, 0},  {},  {},  {} },
    { { 1, 2, 4, 2 }, { 0, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, 2, 1 }, {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {} },
    { { 1, 2, 4, 2 }, { 0, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, 2, 1 }, {0, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {} },
    { { 1, 3, 4, 5, 6 }, { 0, 1, 0, 0, 0 }, { 1, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 }, {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1},  {},  {},  {} },
};

std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
};

Config getConfig() {
    Config config;
    if (CommonTestUtils::vpu::CheckMyriad2()) {
        config[InferenceEngine::MYRIAD_DISABLE_REORDER] = CONFIG_VALUE(YES);
    }
    return config;
}

INSTANTIATE_TEST_CASE_P(StridedSlice_tests, StridedSliceLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(testCases),
                            ::testing::ValuesIn(precisions),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                            ::testing::Values(getConfig())),
                        StridedSliceLayerTest::getTestCaseName);

}  // namespace
