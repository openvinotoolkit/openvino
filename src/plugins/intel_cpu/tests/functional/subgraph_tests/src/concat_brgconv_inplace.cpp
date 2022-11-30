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
 *            param
 *              |
 *            relu
 *              |
 *              /\
 *             /  \
 *           conv1 conv2
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
 * cover: FP32+BF16
 *        conv1: 1x1
 *        conv2: normal
 *        subgraph: in/out stride
 *        concat inplace: brg->concat, concat->concat
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
        const std::vector<size_t> inputShape = {1, 64, 120, 12};
        const InferenceEngine::SizeVector kernel1 = {1, 1};
        const InferenceEngine::SizeVector kernel2 = {3, 3};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin1 = {0, 0};
        const std::vector<ptrdiff_t> padEnd1 = {0, 0};
        const std::vector<ptrdiff_t> padBegin2 = {1, 1};
        const std::vector<ptrdiff_t> padEnd2 = {1, 1};
        const size_t convOutChannels = 80;
        const auto targetFormat = nhwc;
        inPrc = outPrc = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        auto inputParams = ngraph::builder::makeParams(ngPrc, {inputShape});

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
 *          param1       param2
 *            |            |
 *           relu1       relu2
 *           /    \        |
 *         conv1  conv2  conv3
 *           |      |      |
 *           |      |    relu3
 *            \     |     /
 *             concat(inplace)
 *               |
 *             result
 * cover: FP32
 *        conv1: 1x1
 *        conv2: exec_vpad
 *        conv3: exec_base
 *        relu3: perform_outwork
 *        BF16
 *        conv1: 1x1
 *        conv2: exec_trans
 *        conv3: exec_trans
 *        relu3: perform_outwork
 */

class ConcatBrgConvInPlaceTest2 : public testing::WithParamInterface<InferenceEngine::Precision>, public CPUTestsBase,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> obj) {
        std::ostringstream result;
        result << "ConcatBrgConvInPlaceTest2" << obj.param.name();
        return result.str();
    }

    void SetUp() override {
        const std::vector<size_t> inputShape1 = {1, 64, 120, 12};
        const std::vector<size_t> inputShape2 = {1, 64, 119, 11};
        const InferenceEngine::SizeVector kernel1 = {1, 1};
        const InferenceEngine::SizeVector kernel2 = {3, 3};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin1 = {0, 0};
        const std::vector<ptrdiff_t> padEnd1 = {0, 0};
        const std::vector<ptrdiff_t> padBegin2 = {1, 1};
        const std::vector<ptrdiff_t> padEnd2 = {1, 1};
        const std::vector<ptrdiff_t> padBegin3 = {3, 3};
        const std::vector<ptrdiff_t> padEnd3 = {0, 0};
        const size_t convOutChannels = 80;
        const auto targetFormat = nhwc;
        inPrc = outPrc = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        auto inputParams = ngraph::builder::makeParams(ngPrc, {inputShape1, inputShape2});

        auto relu1 = std::make_shared<ngraph::opset3::Relu>(inputParams[0]);
        relu1->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto relu2 = std::make_shared<ngraph::opset3::Relu>(inputParams[1]);
        relu2->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto conv1 = ngraph::builder::makeConvolution(relu1, ngPrc, kernel1, stride, padBegin1,
                                                     padEnd1, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        auto conv2 = ngraph::builder::makeConvolution(relu1, ngPrc, kernel1, stride, padBegin2,
                                                     padEnd2, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        auto conv3 = ngraph::builder::makeConvolution(relu2, ngPrc, kernel2, stride, padBegin3,
                                                     padEnd3, dilation, ngraph::op::PadType::EXPLICIT, convOutChannels);

        auto fusedRelu = std::make_shared<ngraph::opset3::Relu>(conv3);
        auto concat = ngraph::builder::makeConcat(ngraph::OutputVector{conv1, conv2, fusedRelu}, 1);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(concat)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatBrgconvInPlace");
        targetDevice = CommonTestUtils::DEVICE_CPU;
    }
};

// Subgraph:
/*
 *           param1       param2
 *              |           |
 *            relu1       relu2
 *              |           |
 *              /\          |
 *             /  \         |
 *            fq1 fq2      fq3
 *             |   |        |
 *           conv1 conv2  conv3
 *             |   |        |
 *             |   |      relu3
 *             \   |       /
 *             concat(inplace)
 *                    |
 *                  result
 * cover: VNNI I8
 *        conv1: 1x1(use_buffer branch)
 *        conv2: exec_vpad(use_buffer branch)
 *        conv3: exec_base(use_buffer branch)
 *        relu3: perform_outwork(use_buffer branch)
 *        AMX I8
 *        conv1: 1x1
 *        conv2: exec_trans
 *        conv3: exec_trans
 *        relu3: perform_outwork
 */

class ConcatBrgConvInPlaceTest3 : public testing::WithParamInterface<InferenceEngine::Precision>, public CPUTestsBase,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> obj) {
        std::ostringstream result;
        result << "ConcatBrgConvInPlaceTest3" << obj.param.name();
        return result.str();
    }

    void SetUp() override {
        // amx_int8 seems it does not work when input channel > 20480
        const auto extraInChannel = InferenceEngine::with_cpu_x86_avx512_core_amx_int8() ? 0ul : 16ul;
        const std::vector<size_t> inputShape1 = {1ul, 20480ul + extraInChannel, 5ul, 5ul};
        const std::vector<size_t> inputShape2 = {1ul, 20480ul + extraInChannel, 4ul, 4ul};
        const InferenceEngine::SizeVector kernel1 = {1, 1};
        const InferenceEngine::SizeVector kernel2 = {3, 3};
        const InferenceEngine::SizeVector strides = {1, 1};
        const InferenceEngine::SizeVector dilations = {1, 1};
        const std::vector<ptrdiff_t> padsBegin1 = {0, 0};
        const std::vector<ptrdiff_t> padsEnd1 = {0, 0};
        const std::vector<ptrdiff_t> padsBegin2 = {1, 1};
        const std::vector<ptrdiff_t> padsEnd2 = {1, 1};
        const std::vector<ptrdiff_t> padsBegin3 = {3, 3};
        const std::vector<ptrdiff_t> padsEnd3 = {0, 0};
        const size_t convOutChannels = 80;
        const auto targetFormat = nhwc;
        inPrc = outPrc = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        auto inputParams = ngraph::builder::makeParams(ngPrc, {inputShape1, inputShape2});

        auto relu1 = std::make_shared<ngraph::opset3::Relu>(inputParams[0]);
        relu1->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto relu2 = std::make_shared<ngraph::opset3::Relu>(inputParams[1]);
        relu2->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});

        auto data_fake1 = ngraph::builder::makeFakeQuantize(relu1, ngPrc, 255, {1ull}, {-10.f}, {10.f}, {-10.f}, {10.f});
        auto data_fake2 = ngraph::builder::makeFakeQuantize(relu1, ngPrc, 255, {1ull}, {-10.f}, {10.f}, {-10.f}, {10.f});
        auto data_fake3 = ngraph::builder::makeFakeQuantize(relu2, ngPrc, 255, {1ull}, {-10.f}, {10.f}, {-10.f}, {10.f});
        const auto weights1 = ngraph::builder::makeConstant(
            ngraph::element::f32,
            ngraph::Shape{ convOutChannels, inputShape1[1], kernel1[0], kernel1[1] },
            std::vector<float>{}, true);
        const auto weights2 = ngraph::builder::makeConstant(
            ngraph::element::f32,
            ngraph::Shape{ convOutChannels, inputShape1[1], kernel2[0], kernel2[1] },
            std::vector<float>{}, true);
        const auto weights3 = ngraph::builder::makeConstant(
            ngraph::element::f32,
            ngraph::Shape{ convOutChannels, inputShape1[1], kernel2[0], kernel2[1] },
            std::vector<float>{}, true);
        auto weight_fake1 = ngraph::builder::makeFakeQuantize(weights1, ngPrc, 255, {1ull}, {-10.f}, {10.f}, {-10.f}, {10.f});
        auto weight_fake2 = ngraph::builder::makeFakeQuantize(weights2, ngPrc, 255, {1ull}, {-10.f}, {10.f}, {-10.f}, {10.f});
        auto weight_fake3 = ngraph::builder::makeFakeQuantize(weights3, ngPrc, 255, {1ull}, {-10.f}, {10.f}, {-10.f}, {10.f});
        auto conv1 = std::make_shared<ngraph::opset1::Convolution>(
            data_fake1, weight_fake1, strides, padsBegin1, padsEnd1, dilations);
        auto conv2 = std::make_shared<ngraph::opset1::Convolution>(
            data_fake2, weight_fake2, strides, padsBegin2, padsEnd2, dilations);
        auto conv3 = std::make_shared<ngraph::opset1::Convolution>(
            data_fake3, weight_fake3, strides, padsBegin3, padsEnd3, dilations);
        auto fusedRelu = std::make_shared<ngraph::opset3::Relu>(conv3);

        auto concat = ngraph::builder::makeConcat(ngraph::OutputVector{conv1, conv2, fusedRelu}, 1);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(concat)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatBrgconvInPlace");
        targetDevice = CommonTestUtils::DEVICE_CPU;
    }
};

namespace {
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
    ConcatBrgConvInPlaceTest1::getTestCaseName);

TEST_P(ConcatBrgConvInPlaceTest2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!InferenceEngine::with_cpu_x86_avx512_core_vnni())
        GTEST_SKIP();
    if (this->GetParam() == Precision::BF16 && !InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    Run();

    if (this->GetParam() == Precision::BF16) {
        selectedType = "unknown_BF16";
    } else {
        selectedType = "unknown_FP32";
    }

    CheckNumberOfNodesWithType(executableNetwork, "Reorder", 6);
    CheckPluginRelatedResults(executableNetwork, "Concatenation");
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatBrgConvInPlaceTest2_CPU,
                         ConcatBrgConvInPlaceTest2,
                         testing::Values(Precision::FP32, Precision::BF16),
                         ConcatBrgConvInPlaceTest2::getTestCaseName);

TEST_P(ConcatBrgConvInPlaceTest3, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!InferenceEngine::with_cpu_x86_avx512_core_vnni())
        GTEST_SKIP();
    if (this->GetParam() == Precision::BF16 && !InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    Run();

    if (this->GetParam() == Precision::BF16) {
        selectedType = "unknown_BF16";
    } else if (this->GetParam() == Precision::FP32) {
        selectedType = "unknown_FP32";
    } else {
        selectedType = "unknown_I8";
    }

    CheckNumberOfNodesWithType(executableNetwork, "Reorder", 6);
    CheckPluginRelatedResults(executableNetwork, "Concatenation");
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatBrgConvInPlaceTest3_CPU,
                         ConcatBrgConvInPlaceTest3,
                         testing::Values(Precision::FP32),
                         ConcatBrgConvInPlaceTest3::getTestCaseName);
}// namespace
} // namespace SubgraphTestsDefinitions