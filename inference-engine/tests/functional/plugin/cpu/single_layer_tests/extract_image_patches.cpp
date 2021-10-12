// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include <shared_test_classes/single_layer/extract_image_patches.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
using LayerTestsDefinitions::extractImagePatchesTuple;

typedef std::tuple<
    extractImagePatchesTuple,
    CPUSpecificParams> extractImagePatchesCPUTestParamsSet;

class ExtractImagePatchesLayerCPUTest : public testing::WithParamInterface<extractImagePatchesCPUTestParamsSet>,
    virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<extractImagePatchesCPUTestParamsSet> obj) {
        extractImagePatchesTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ExtractImagePatchesTest::getTestCaseName(testing::TestParamInfo<extractImagePatchesTuple>(
            basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        extractImagePatchesTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<size_t> inputShape, kernel, strides, rates;
        ngraph::op::PadType pad_type;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, kernel, strides, rates, pad_type, netPrecision, inPrc, outPrc, inLayout, targetDevice) = basicParamsSet;
        selectedType = std::string("ref_any_") + netPrecision.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto inputNode = std::make_shared<ngraph::opset6::Parameter>(ngPrc, ngraph::Shape(inputShape));
        ngraph::ParameterVector params = {inputNode};

        auto extImgPatches = std::make_shared<ngraph::opset6::ExtractImagePatches>(
                inputNode, ngraph::Shape(kernel), ngraph::Strides(strides), ngraph::Shape(rates), pad_type);
        ngraph::ResultVector results{std::make_shared<ngraph::opset6::Result>(extImgPatches)};
        function = std::make_shared<ngraph::Function>(results, params, "ExtractImagePatches");
    }
};

TEST_P(ExtractImagePatchesLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "ExtractImagePatches");
}

namespace {
    const std::vector<std::vector<size_t>> inShapes = {{2, 3, 13, 37}};
    const std::vector<std::vector<size_t>> kSizes = {{1, 5}, {3, 4}, {3, 1}};
    const std::vector<std::vector<size_t>> strides = {{1, 2}, {2, 2}, {2, 1}};
    const std::vector<std::vector<size_t>> rates = {{1, 3}, {3, 3}, {3, 1}};

    const std::vector<ngraph::op::PadType> autoPads = {ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER};
    const std::vector<Precision> netPrecision = {Precision::I8, Precision::BF16, Precision::FP32};
    const CPUSpecificParams CPUParams = emptyCPUSpec;

const auto Layer_params = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(kSizes),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(rates),
        ::testing::ValuesIn(autoPads),
        ::testing::ValuesIn(netPrecision),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_ExtractImagePatches_CPU, ExtractImagePatchesLayerCPUTest,
                        ::testing::Combine(Layer_params, ::testing::Values(CPUParams)),
                        ExtractImagePatchesLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
