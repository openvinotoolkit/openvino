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
    int pooledH;
    int pooledW;
    float spatialScale;
    int samplingRatio;
    std::pair<std::vector<float>, std::vector<size_t>> proposal;
    std::string mode;
    std::vector<size_t> inputShape;
}  // namespace

typedef std::tuple<
        int,                                                 // bin's column count
        int,                                                 // bin's row count
        float,                                               // scale for given region considering actual input size
        int,                                                 // pooling ratio
        std::pair<std::vector<float>, std::vector<size_t>>,  // united proposal vector of coordinates and indexes
        std::string,                                         // pooling mode
        std::vector<size_t>                                  // feature map shape
> ROIAlignSpecificParams;

typedef std::tuple<
        ROIAlignSpecificParams,
        InferenceEngine::Precision,     // Net precision
        LayerTestsUtils::TargetDevice   // Device name
> ROIAlignLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::ROIAlignLayerTestParams,
        CPUSpecificParams> ROIAlignLayerCPUTestParamsSet;

class ROIAlignLayerCPUTest : public testing::WithParamInterface<ROIAlignLayerCPUTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIAlignLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::ROIAlignLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        Precision netPr;
        ROIAlignSpecificParams roiPar;
        std::tie(roiPar, netPr, td) = basicParamsSet;
        std::tie(pooledH, pooledW, spatialScale, samplingRatio,
                 proposal, mode, inputShape) = roiPar;
        std::ostringstream result;
        result << "ROIAlignTest_";
        result << std::to_string(obj.index);
        result << "pooledH=" << pooledH << "_";
        result << "pooledW=" << pooledW << "_";
        result << "spatialScale=" << spatialScale << "_";
        result << "samplingRatio=" << samplingRatio << "_";
        result << (netPr == Precision::FP32 ? "FP32" : "BF16") << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::ROIAlignLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::ROIAlignSpecificParams roiAlignParams;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(roiAlignParams, netPrecision, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        std::tie(pooledH, pooledW, spatialScale, samplingRatio,
                 proposal, mode, inputShape) = roiAlignParams;

        std::vector<float> proposalVector = proposal.first;
        std::vector<size_t> roiIdxVector = proposal.second;

        ngraph::Shape coordsShape = { proposalVector.size() / 4, 4 };
        ngraph::Shape idxVectorShape = { roiIdxVector.size() };

        auto roisIdx = ngraph::builder::makeConstant<size_t>(ngraph::element::i32, idxVectorShape, roiIdxVector);
        auto coords = ngraph::builder::makeConstant<float>(ngraph::element::f32, coordsShape, proposalVector);
        auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});

        auto roialign = std::make_shared<ngraph::opset3::ROIAlign>(params[0], coords, roisIdx, pooledH, pooledW,
                                                                   samplingRatio, spatialScale, mode);
        roialign->get_rt_info() = getCPUInfo();
        selectedType = std::string("unknown_") + inPrc.name();

        threshold = 1e-2;
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roialign)};
        function = std::make_shared<ngraph::Function>(results, params, "ROIAlign");
    }
};

TEST_P(ROIAlignLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "ROIAlign");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {}, {}});
    resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {}, {}});
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, nc, x}, {nChw16c}, {}, {}});
    } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc, x}, {nChw8c}, {}, {}});
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16
};

const std::vector<int> spatialBinXVector = { 2 };

const std::vector<int> spatialBinYVector = { 2 };

const std::vector<float> spatialScaleVector = { 1.0f };

const std::vector<int> poolingRatioVector = { 7 };

const std::vector<std::string> modeVector = {
        "avg",
        "max"
};

const std::vector<std::vector<size_t>> inputShapeVector = {
        SizeVector({ 2, 18, 20, 20 }),
        SizeVector({ 2, 4, 20, 20 }),
        SizeVector({ 2, 4, 20, 40 }),
        SizeVector({ 10, 1, 20, 20 })
};


const std::vector<std::pair<std::vector<float>, std::vector<size_t>>> propVector = {
        {{ 1, 1, 19, 19, 1, 1, 19, 19, }, { 0, 1 }},
        {{ 1, 1, 19, 19 }, { 1 }}
};

const auto roiAlignParams = ::testing::Combine(
        ::testing::ValuesIn(spatialBinXVector),       // bin's column count
        ::testing::ValuesIn(spatialBinYVector),       // bin's row count
        ::testing::ValuesIn(spatialScaleVector),      // scale for given region considering actual input size
        ::testing::ValuesIn(poolingRatioVector),      // pooling ratio for bin
        ::testing::ValuesIn(propVector),              // united vector of coordinates and batch id's
        ::testing::ValuesIn(modeVector),              // pooling mode
        ::testing::ValuesIn(inputShapeVector)         // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlignLayoutTest, ROIAlignLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        roiAlignParams,
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                ::testing::ValuesIn(filterCPUInfoForDevice())),
                ROIAlignLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
