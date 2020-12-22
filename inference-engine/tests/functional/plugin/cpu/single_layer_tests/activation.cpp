// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/activation.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
        LayerTestsDefinitions::activationParams,
        CPUSpecificParams>
        ActivationLayerCPUTestParamSet;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj) {
        LayerTestsDefinitions::activationParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ActivationLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::activationParams>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 15, 0, 32768);
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::activationParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::Precision netPrecision;
        std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
        std::pair<ActivationTypes, std::vector<float>> activationDecl;
        std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = basicParamsSet;
        selectedType = getPrimitiveType() + "_" + inPrc.name();

        activationType = activationDecl.first;
        auto constantsValue = activationDecl.second;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
        auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, shapes.second, constantsValue);
        activation->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params, "Activation");
    }
};

TEST_P(ActivationLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Eltwise");
}


namespace {
// list only types supported by eltwise
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sqrt,        {{}}},
        {Sigmoid,     {{}}},
        {Tanh,        {{}}},
        {Relu,        {{}}},
        {Gelu,        {{}}},
        {Exp,         {{}}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Elu,         {{0.1f}}},
        {Swish,       {{0.1f}}},
        {HSwish,      {{}}},
        {Mish,        {{}}},
        {PReLu, {{-0.01f}}}
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic4D = {
        {{2, 4, 4, 1}, {{}}},
        {{2, 17, 5, 4}, {{}}},
};

std::vector<Precision> bf16InpOutPrc = {Precision::BF16, Precision::FP32};

const auto basicCases4D = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
            ::testing::Values(Precision::BF16),
            ::testing::ValuesIn(bf16InpOutPrc),
            ::testing::ValuesIn(bf16InpOutPrc),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::ValuesIn(CommonTestUtils::combineParams(basic4D)),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D))
);

INSTANTIATE_TEST_CASE_P(smoke_Activation4D_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic5D = {
        {{2, 4, 3, 4, 1}, {{}}},
        {{2, 17, 7, 5, 4}, {{}}},
};

const auto basicCases5D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
                ::testing::Values(Precision::BF16),
                ::testing::ValuesIn(bf16InpOutPrc),
                ::testing::ValuesIn(bf16InpOutPrc),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(CommonTestUtils::combineParams(basic5D)),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D))
);

INSTANTIATE_TEST_CASE_P(smoke_Activation5D_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions