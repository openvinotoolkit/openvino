// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/myriad_common_test_utils.hpp"
#include <vpu/private_plugin_config.hpp>

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

typedef std::map<std::string, std::string> Config;

std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{2}},
        {{1, 1, 1, 3}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{16, 16, 96}, {96}},
        {{52, 1, 52, 3, 2}, {2}}
};

std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::ADD
};

Config getConfig() {
    Config config;
    config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    if (CommonTestUtils::vpu::CheckMyriad2()) {
        config[InferenceEngine::MYRIAD_DISABLE_REORDER] = CONFIG_VALUE(YES);
    }
    return config;
}

const auto multiply_params = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(getConfig()));

INSTANTIATE_TEST_CASE_P(CompareWithRefs, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);

}  // namespace
