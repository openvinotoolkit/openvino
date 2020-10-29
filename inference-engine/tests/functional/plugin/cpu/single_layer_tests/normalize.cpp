// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/normalize_l2.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::NormalizeL2LayerTestParams,
        CPUSpecificParams,
        Precision, // CNNNetwork input precision
        Precision> // CNNNetwork output precision
NormalizeL2LayerCPUTestParamSet;

class NormalizeL2LayerCPUTest : public testing::WithParamInterface<NormalizeL2LayerCPUTestParamSet>,
                        virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2LayerCPUTestParamSet> obj) {
        LayerTestsDefinitions::NormalizeL2LayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        Precision inputPrecision, outputPrecision;
        std::tie(basicParamsSet, cpuParams, inputPrecision, outputPrecision) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::NormalizeL2LayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::NormalizeL2LayerTestParams>(
                basicParamsSet, 0));

        result << "_" << "CNNInpPrc=" << inputPrecision.name();
        result << "_" << "CNNOutPrc=" << outputPrecision.name();

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        LayerTestsDefinitions::NormalizeL2LayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams, inPrc, outPrc) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        // Withing the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
        configuration.insert({PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO});

        std::vector<int64_t> axes;
        float eps;
        ngraph::op::EpsMode eps_mode;
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision  netPrecision;
        std::tie(axes, eps, eps_mode, inputShapes, netPrecision, targetDevice) = basicParamsSet;
        auto netPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto param = ngraph::builder::makeParams(netPrc, {inputShapes});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
        auto normalize_l2 = ngraph::builder::makeNormalizeL2(paramOuts[0], axes, eps, eps_mode);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(normalize_l2)};

        std::string strExpectedPrc;
        if (Precision::BF16 == inPrc) {
            strExpectedPrc = "BF16";
        } else if (Precision::FP32 == inPrc) {
            strExpectedPrc = "FP32";
        }

        std::string isaType;
        if (with_cpu_x86_avx512f()) {
            isaType = "unknown";
        } else if (with_cpu_x86_avx2()) {
            isaType = "jit_avx2";
        } else if (with_cpu_x86_sse42()) {
            isaType = "jit_sse42";
        } else {
            isaType = "ref";
        }
        selectedType = isaType + "_" + strExpectedPrc;

        threshold = 0.015f; //slightly increase threshold

        normalize_l2->get_rt_info() = getCPUInfo();

        function = std::make_shared<ngraph::Function>(results, param, "Normalize");
    }
};

TEST_P(NormalizeL2LayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Normalize");
}

namespace {

const std::vector<std::vector<int64_t>> axes = {
        {},
        {1},
};
const std::vector<float> eps = { 1e-4f };

const std::vector<ngraph::op::EpsMode> epsMode = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX,
};

std::vector<Precision> inpOutPrc = {Precision::BF16};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};


const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const auto NormalizeL23D = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes),
            testing::ValuesIn(eps),
            testing::ValuesIn(epsMode),
            testing::Values(std::vector<size_t>{1, 32, 17}),
            testing::ValuesIn(inpOutPrc),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(inpOutPrc),
        testing::ValuesIn(inpOutPrc));

INSTANTIATE_TEST_CASE_P(NormalizeL2CompareWithRefs_3D, NormalizeL2LayerCPUTest, NormalizeL23D, NormalizeL2LayerCPUTest::getTestCaseName);

const auto NormalizeL24D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(eps),
                testing::ValuesIn(epsMode),
                testing::Values(std::vector<size_t>{1, 3, 10, 5}),
                testing::ValuesIn(inpOutPrc),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(inpOutPrc),
        testing::ValuesIn(inpOutPrc));

INSTANTIATE_TEST_CASE_P(NormalizeL2CompareWithRefs_4D, NormalizeL2LayerCPUTest, NormalizeL24D, NormalizeL2LayerCPUTest::getTestCaseName);


} // namespace
} // namespace CPULayerTestsDefinitions
