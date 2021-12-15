// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<float> proposal;
    std::vector<size_t> featureMapShape;
    size_t spatialBinsX;
    size_t spatialBinsY;
    float spatialScale;
    size_t groupSize;
    size_t outputDim;
    std::string mode;
}  // namespace

typedef std::tuple<
        std::vector<size_t>,            // feature map shape
        std::vector<float>,             // coords shape
        size_t,                         // output_dim
        size_t,                         // group_size
        float,                          // Spatial scale
        size_t,                         // spatial_bins_x
        size_t,                         // spatial_bins_y
        std::string                     // mode
> PSROIPoolingSpecificParams;

typedef std::tuple<
        PSROIPoolingSpecificParams,
        InferenceEngine::Precision,     // Net precision
        LayerTestsUtils::TargetDevice   // Device name
> PSROIPoolingLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::PSROIPoolingLayerTestParams,
        CPUSpecificParams> PSROIPoolingLayerCPUTestParamsSet;

class PSROIPoolingLayerCPUTest : public testing::WithParamInterface<PSROIPoolingLayerCPUTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PSROIPoolingLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::PSROIPoolingLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        Precision netPr;
        PSROIPoolingSpecificParams psroiPar;
        std::tie(psroiPar, netPr, td) = basicParamsSet;
        std::tie(featureMapShape, proposal, outputDim, groupSize,
                 spatialScale, spatialBinsX, spatialBinsY, mode) = psroiPar;
        std::ostringstream result;
        result << "PSROIPoolingTest_";
        result << std::to_string(obj.index) << "_";
        result << "binsX=" << spatialBinsX << "_";
        result << "binsY=" << spatialBinsY << "_";
        result << "spatialScale=" << spatialScale << "_";
        result << "outputD=" << outputDim << "_";
        result << "groupS=" << groupSize << "_";
        result << netPr.name() << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::PSROIPoolingLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::PSROIPoolingSpecificParams psroiPoolingParams;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(psroiPoolingParams, netPrecision, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        std::tie(featureMapShape, proposal, outputDim, groupSize,
                 spatialScale, spatialBinsX, spatialBinsY, mode) = psroiPoolingParams;


        ngraph::Shape proposalShape = { proposal.size() / 5, 5 };

        auto coords = ngraph::builder::makeConstant<float>(ngraph::element::f32, proposalShape, proposal);
        auto params = ngraph::builder::makeParams(ngraph::element::f32, {featureMapShape});

        auto psroi = std::make_shared<ngraph::op::v0::PSROIPooling>(params[0], coords, outputDim, groupSize,
                                                       spatialScale, spatialBinsX, spatialBinsY, mode);
        psroi->get_rt_info() = getCPUInfo();
        selectedType = getPrimitiveType() + "_" + inPrc.name();

        threshold = 1e-2;
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(psroi)};
        function = std::make_shared<ngraph::Function>(results, params, "PSROIPooling");
    }
};

TEST_P(PSROIPoolingLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "PSROIPooling");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> resCPUParams {
    CPUSpecificParams{{nchw, nc}, {nchw}, {}, {}},
    CPUSpecificParams{{nhwc, nc}, {nhwc}, {}, {}},
    CPUSpecificParams{{nChw16c, nc}, {nChw16c}, {}, {}}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16
};

const std::vector<float> spatialScaleVector = { 1.0f };

const std::vector<std::vector<size_t>> inputShapeVector = {
        SizeVector({ 2, 200, 20, 20 }),
        SizeVector({ 2, 200, 20, 16 }),
        SizeVector({ 2, 200, 16, 20 }),
        SizeVector({ 3, 200, 16, 16 })
};

const std::vector<std::vector<float>> averagePropVector = {
        { 0, 0.9, 0.9, 18.9, 18.9,
          1, 0.9, 0.9, 18.9, 18.9 },
        { 1, 1, 1, 15, 15 }
};

const std::vector<std::vector<float>> bilinearPropVector = {
        { 0, 0.1, 0.1, 0.9, 0.9,
          1, 0.1, 0.1, 0.9, 0.9 },
        { 1, 0.1, 0.1, 0.9, 0.9 }
};

const auto psroiPoolingAverageParams = ::testing::Combine(
        ::testing::ValuesIn(inputShapeVector),
        ::testing::ValuesIn(averagePropVector),
        ::testing::Values(50),
        ::testing::Values(2),
        ::testing::ValuesIn(spatialScaleVector),
        ::testing::Values(1),
        ::testing::Values(1),
        ::testing::Values("average")
);

const auto psroiPoolingBilinearParams = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{3, 32, 20, 20}),
        ::testing::ValuesIn(bilinearPropVector),
        ::testing::Values(4),
        ::testing::Values(3),
        ::testing::ValuesIn(spatialScaleVector),
        ::testing::Values(4),
        ::testing::Values(2),
        ::testing::Values("bilinear")
);

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest, PSROIPoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        psroiPoolingAverageParams,
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUSpecificParams(resCPUParams))),
                        PSROIPoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingBilinearLayoutTest, PSROIPoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        psroiPoolingBilinearParams,
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUSpecificParams(resCPUParams))),
                        PSROIPoolingLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
