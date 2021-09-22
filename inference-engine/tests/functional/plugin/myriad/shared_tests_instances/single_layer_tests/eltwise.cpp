// Copyright (C) 2018-2021 Intel Corporation
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

std::vector<InferenceEngine::Precision> fpTypes = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

std::vector<InferenceEngine::Precision> intTypes = {
        InferenceEngine::Precision::I32,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseMathTypesFP = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        ngraph::helpers::EltwiseTypes::POWER,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseMathTypesINT = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::DIVIDE,
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseMathFP,
                        EltwiseLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::ValuesIn(eltwiseMathTypesFP),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(opTypes),
                            ::testing::ValuesIn(fpTypes),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                            ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        EltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseMathInt,
                        EltwiseLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(eltwiseMathTypesINT),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::ValuesIn(opTypes),
                                ::testing::ValuesIn(intTypes),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        EltwiseLayerTest::getTestCaseName);

}  // namespace
