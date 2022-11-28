// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
// Subgraph:
/*
 *           paramter
 *              |
 *            reLu
 *              |
 *              /\
 *             /  \
 *           conv conv
 *             |   |
 *              \ /
 *            concat(inplace)
 *               /\
 *              /  \
 *         multiply \
 *        (subgraph)/
 *              \  /
 *             concat(inplace)
 *               |
 *             result
 */

class ConcatBrgConvInPlaceTest1 : public testing::WithParamInterface<InferenceEngine::Precision>, public CPUTestsBase,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> obj) {
        std::ostringstream result;
        result << "ConcatBrgConvInPlaceTest1" << obj.param.name();
        return result.str();
    }

    void SetUp() override {
        const std::vector<size_t> inputShape = {1, 64, 120, 120};
        const InferenceEngine::SizeVector kernel1 = {1, 1};
        const InferenceEngine::SizeVector kernel2 = {3, 3};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin1 = {0, 0};
        const std::vector<ptrdiff_t> padEnd1 = {0, 0};
        const std::vector<ptrdiff_t> padBegin2 = {1, 1};
        const std::vector<ptrdiff_t> padEnd2 = {1, 1};
        const size_t convOutChannels = 64;
        const auto targetFormat = nhwc;
        inPrc = outPrc = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        auto inputParams = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});

        auto relu = std::make_shared<ngraph::opset3::Relu>(inputParams[0]);
        relu->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto conv1 = ngraph::builder::makeConvolution(relu, ngPrc, kernel1, stride, padBegin1,
                                                     padEnd1, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        auto conv2 = ngraph::builder::makeConvolution(relu, ngPrc, kernel2, stride, padBegin2,
                                                     padEnd2, dilation, ngraph::op::PadType::AUTO, convOutChannels);

        auto concat1 = ngraph::builder::makeConcat(ngraph::OutputVector{conv1, conv2}, 1);
        auto mulConst = ngraph::builder::makeConstant(ngPrc, {1, convOutChannels * 2, 1, 1}, std::vector<float>{}, true);
        auto mul = std::make_shared<ngraph::opset3::Multiply>(concat1, mulConst);
        auto concat2 = ngraph::builder::makeConcat(ngraph::OutputVector{concat1, mul}, 1);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(concat2)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatBrgconvInPlace");
        targetDevice = CommonTestUtils::DEVICE_CPU;
    }
};

// Subgraph:
/*
 *           paramter
 *              |
 *            reLu
 *              |
 *              /\
 *             /  \
 *           conv conv
 *             |   |
 *              \ /
 *            concat(inplace)
 *               \  conv
 *                \  /
 *                 \/
 *                concat(inplace)
 *                  |
 *                result
 */

class ConcatBrgConvInPlaceTest : public testing::WithParamInterface<InferenceEngine::Precision>, public CPUTestsBase,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> obj) {
        std::ostringstream result;
        result << "ConcatBrgConvInPlaceTest" << obj.param.name();
        return result.str();
    }

    void SetUp() override {
        const std::vector<size_t> inputShape = {1, 64, 120, 120};
        const InferenceEngine::SizeVector kernel1 = {1, 1};
        const InferenceEngine::SizeVector kernel2 = {3, 3};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin1 = {0, 0};
        const std::vector<ptrdiff_t> padEnd1 = {0, 0};
        const std::vector<ptrdiff_t> padBegin2 = {1, 1};
        const std::vector<ptrdiff_t> padEnd2 = {1, 1};
        const size_t convOutChannels = 64;
        const auto targetFormat = nhwc;
        inPrc = outPrc = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        auto inputParams = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});

        auto relu = std::make_shared<ngraph::opset3::Relu>(inputParams[0]);
        relu->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto conv1 = ngraph::builder::makeConvolution(relu, ngPrc, kernel1, stride, padBegin1,
                                                     padEnd1, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        auto conv2 = ngraph::builder::makeConvolution(relu, ngPrc, kernel2, stride, padBegin2,
                                                     padEnd2, dilation, ngraph::op::PadType::AUTO, convOutChannels);

        auto concat1 = ngraph::builder::makeConcat(ngraph::OutputVector{conv1, conv2}, 1);
        auto conv3 = ngraph::builder::makeConvolution(relu, ngPrc, kernel1, stride, padBegin1,
                                                     padEnd1, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        auto concat2 = ngraph::builder::makeConcat(ngraph::OutputVector{concat1, conv3}, 1);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(concat2)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatBrgconvInPlace");
        targetDevice = CommonTestUtils::DEVICE_CPU;
    }
};

namespace {
TEST_P(ConcatBrgConvInPlaceTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!InferenceEngine::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    if (this->GetParam() == Precision::BF16 && !InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    Run();

    if (this->GetParam() == Precision::BF16) {
        selectedType = "unknown_BF16";
    } else {
        selectedType = "unknown_FP32";
    }
    CheckNumberOfNodesWithType(executableNetwork, "Reorder", 5);
    CheckPluginRelatedResults(executableNetwork, "Concatenation");
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatBrgConvInPlaceTest_CPU, ConcatBrgConvInPlaceTest,
    testing::Values(Precision::FP32, Precision::BF16),
    ConcatBrgConvInPlaceTest::getTestCaseName);

TEST_P(ConcatBrgConvInPlaceTest1, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!InferenceEngine::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    if (this->GetParam() == Precision::BF16 && !InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    Run();

    if (this->GetParam() == Precision::BF16) {
        selectedType = "unknown_BF16";
    } else {
        selectedType = "unknown_FP32";
    }
    CheckNumberOfNodesWithType(executableNetwork, "Reorder", 5);
    CheckPluginRelatedResults(executableNetwork, "Concatenation");
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatBrgConvInPlaceTest1_CPU, ConcatBrgConvInPlaceTest1,
    testing::Values(Precision::FP32, Precision::BF16),
    ConcatBrgConvInPlaceTest::getTestCaseName);

}// namespace
} // namespace SubgraphTestsDefinitions