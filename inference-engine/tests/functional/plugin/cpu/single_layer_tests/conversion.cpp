// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/conversion.hpp>

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace CPULayerTestsDefinitions  {

class ConvertCPULayerTest : public ConversionLayerTest {};

TEST_P(ConvertCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ConversionParamsTuple params = GetParam();
    inPrc = std::get<2>(params);
    outPrc = std::get<3>(params);

    Run();
}

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
    ngraph::helpers::ConversionTypes::CONVERT,
    ngraph::helpers::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

// List of precisions natively supported by mkldnn.
const std::vector<Precision> precisions = {
        Precision::U8,
        Precision::I8,
        Precision::I32,
        Precision::FP32,
        Precision::BF16
};

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest_From_BF16, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(conversionOpTypes),
                                ::testing::Values(inShape),
                                ::testing::Values(Precision::BF16),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest_To_BF16, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(conversionOpTypes),
                                ::testing::Values(inShape),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConversionLayerTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
