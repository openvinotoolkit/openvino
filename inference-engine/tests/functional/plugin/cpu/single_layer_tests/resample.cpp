// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "../src/legacy_api/include/legacy/ngraph_ops/interp.hpp"
#include "ngraph/type/bfloat16.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using ResampleLayerTestParams = std::tuple<
    float,                              // factor
    bool,                               // antialias
    std::string,                        // mode
    InferenceEngine::SizeVector,        // inputShape
    InferenceEngine::Precision,         // netPrecision
    std::string                         // targetDevice
>;

typedef std::tuple<
        ResampleLayerTestParams,
        CPUSpecificParams>
ResampleLayerCPUTestParamSet;

class ResampleLayerCPUTest : public testing::WithParamInterface<ResampleLayerCPUTestParamSet>,
    virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName1(testing::TestParamInfo<ResampleLayerTestParams> obj) {
        std::vector<int64_t> axes;
        float factor;
        bool antialias;
        std::string mode;
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(factor, antialias, mode, inputShape, netPrecision, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "factor=" << factor << "_";
        result << "antialias=" << antialias << "_";
        result << "mode=" << mode << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    static std::string getTestCaseName(testing::TestParamInfo<ResampleLayerCPUTestParamSet> obj) {
        ResampleLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << getTestCaseName1(testing::TestParamInfo<ResampleLayerTestParams>(basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        ResampleLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        // Withing the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
        configuration.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO });

        ngraph::op::ResampleIEAttrs resampleAttrs;
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision  netPrecision;
        std::tie(resampleAttrs.factor, resampleAttrs.antialias, resampleAttrs.mode, inputShapes, netPrecision, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        auto netPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto param = ngraph::builder::makeParams(netPrc, { inputShapes });
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));

        const auto interpolateShape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 1 },
            std::vector<int64_t>({ static_cast<int64_t>(inputShapes[1]) }));
        const auto resample = std::make_shared<ngraph::op::ResampleV2>(paramOuts[0], interpolateShape, resampleAttrs);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(resample) };

        if (Precision::BF16 == netPrecision) {
            selectedType = "unknown_BF16";
        } else if (Precision::FP32 == netPrecision) {
            selectedType = "unknown_FP32";
        }

        threshold = 0.015f;

        resample->get_rt_info() = getCPUInfo();

        function = std::make_shared<ngraph::Function>(results, param, "Resample");
    }
};

TEST_P(ResampleLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Resample");
}

namespace {

const std::vector<std::vector<size_t>> inputShapes_4D = {
    {2, 16, 10, 20},
    {2, 3, 15, 25},
    {2, 3, 10, 20}
};

const std::vector<std::vector<size_t>> inputShapes_5D = {
    {2, 3, 8, 15, 5},
    {2, 2, 7, 5, 10}
};

const std::vector<float> factor = {
    1.f,
    0.25f,
    4.f
};

const std::vector<bool> antialias = {
    true,
    false
};

const std::vector<Precision> netPrecisions = {
    Precision::FP32,
    Precision::BF16
};

std::vector<CPUSpecificParams> cpuParams_4D = {
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
    CPUSpecificParams({nhwc}, {nhwc}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
    CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
    CPUSpecificParams({ndhwc}, {ndhwc}, {}, {})
};

const auto Resample4Dlinear = testing::Combine(
    testing::Combine(
        testing::ValuesIn(factor),
        testing::ValuesIn(antialias),
        testing::Values("linear"),
        testing::ValuesIn(inputShapes_4D),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)),
    testing::Values(CPUSpecificParams({ nchw }, { nchw }, {}, {})));

INSTANTIATE_TEST_CASE_P(smoke_ResampleCompareWithRefs_4Dlinear, ResampleLayerCPUTest, Resample4Dlinear, ResampleLayerCPUTest::getTestCaseName);

const auto Resample5Dlinear = testing::Combine(
    testing::Combine(
        testing::ValuesIn(factor),
        testing::ValuesIn(antialias),
        testing::Values("linear"),
        testing::ValuesIn(inputShapes_5D),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)),
    testing::Values(CPUSpecificParams({ ncdhw }, { ncdhw }, {}, {})));

INSTANTIATE_TEST_CASE_P(smoke_ResampleCompareWithRefs_5Dlinear, ResampleLayerCPUTest, Resample5Dlinear, ResampleLayerCPUTest::getTestCaseName);

const auto Resample4Dnearest = testing::Combine(
    testing::Combine(
        testing::ValuesIn(factor),
        testing::Values(false),
        testing::Values("nearest"),
        testing::ValuesIn(inputShapes_4D),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)),
    testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)));

INSTANTIATE_TEST_CASE_P(smoke_ResampleCompareWithRefs_4Dnearest, ResampleLayerCPUTest, Resample4Dnearest, ResampleLayerCPUTest::getTestCaseName);

const auto Resample5Dnearest = testing::Combine(
    testing::Combine(
        testing::ValuesIn(factor),
        testing::Values(false),
        testing::Values("nearest"),
        testing::ValuesIn(inputShapes_5D),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)),
    testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)));

INSTANTIATE_TEST_CASE_P(smoke_ResampleCompareWithRefs_5Dnearest, ResampleLayerCPUTest, Resample5Dnearest, ResampleLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
