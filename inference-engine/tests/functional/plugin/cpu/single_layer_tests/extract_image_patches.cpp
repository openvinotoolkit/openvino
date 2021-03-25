// Copyright (C) 2018-2021 Intel Corporation
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
//using LayerTestsDefinitions::convLayerTestParamsSet;

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

        if (inPrc == Precision::FP32) {
            selectedType = std::string("unknown_") + Precision(Precision::FP32).name();
        } else if (inPrc == Precision::I16) {
            selectedType = std::string("unknown_") + Precision(Precision::I32).name();
        } else if (inPrc == Precision::U8) {
            selectedType = std::string("unknown_") + Precision(Precision::I8).name();
        }
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
    const std::vector<ngraph::op::PadType> autoPads = {ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER};
    const std::vector<Precision> inPrecision = {Precision::FP32, Precision::I16, Precision::U8};
    const std::vector<CPUSpecificParams> CPUParams = {{{}, {}, {"ref_any"}, "ref_any"},
                                                      {{}, {}, {"jit_sse42"}, "jit_sse42"},
                                                      {{}, {}, {"jit_avx2"}, "jit_avx2"},
                                                      {{}, {}, {"jit_avx512"}, "jit_avx512"}};
/* ============= 1D ============= */
const auto Layer_params_1D = ::testing::Combine(
        ::testing::Values(std::vector<size_t> {1, 1, 1, 37}),   // InShapes
        ::testing::Values(std::vector<size_t> {1, 5}),          // Kernel Sizes
        ::testing::Values(std::vector<size_t> {1, 2}),          // Strides
        ::testing::Values(std::vector<size_t> {1, 3}),          // Rates
        ::testing::ValuesIn(autoPads),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(inPrecision),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches_CPU_1D, ExtractImagePatchesLayerCPUTest,
                        ::testing::Combine(Layer_params_1D, ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams))),
                        ExtractImagePatchesLayerCPUTest::getTestCaseName);

/* ============= 2D ============= */
const auto Layer_params_2D = ::testing::Combine(
        ::testing::Values(std::vector<size_t> {1, 1, 13, 17}),   // InShape
        ::testing::Values(std::vector<size_t> {3, 4}),          // Kernel Size
        ::testing::Values(std::vector<size_t> {2, 2}),          // Strides
        ::testing::Values(std::vector<size_t> {3, 3}),          // Rates
        ::testing::ValuesIn(autoPads),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(inPrecision),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches_CPU_2D, ExtractImagePatchesLayerCPUTest,
                        ::testing::Combine(Layer_params_2D, ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams))),
                        ExtractImagePatchesLayerCPUTest::getTestCaseName);

/* ============= 3D ============= */
const auto Layer_params_3D = ::testing::Combine(
        ::testing::Values(std::vector<size_t> {1, 3, 7, 11}),   // InShape
        ::testing::Values(std::vector<size_t> {2, 3}),          // Kernel Size
        ::testing::Values(std::vector<size_t> {1, 2}),          // Strides
        ::testing::Values(std::vector<size_t> {2, 2}),          // Rates
        ::testing::ValuesIn(autoPads),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(inPrecision),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches_CPU_3D, ExtractImagePatchesLayerCPUTest,
                        ::testing::Combine(Layer_params_3D, ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams))),
                        ExtractImagePatchesLayerCPUTest::getTestCaseName);

/* ============= 4D ============= */
const auto Layer_params_4D = ::testing::Combine(
        ::testing::Values(std::vector<size_t> {4, 2, 9, 5}),   // InShape
        ::testing::Values(std::vector<size_t> {3, 1}),          // Kernel Size
        ::testing::Values(std::vector<size_t> {1, 3}),          // Strides
        ::testing::Values(std::vector<size_t> {2, 1}),          // Rates
        ::testing::ValuesIn(autoPads),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(inPrecision),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches_CPU_4D, ExtractImagePatchesLayerCPUTest,
                        ::testing::Combine(Layer_params_4D, ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams))),
                        ExtractImagePatchesLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
