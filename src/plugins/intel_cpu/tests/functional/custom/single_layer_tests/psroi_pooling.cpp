// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
typedef std::tuple<ov::Shape,           // feature map shape
                   std::vector<float>,  // coords shape
                   size_t,              // output_dim
                   size_t,              // group_size
                   float,               // Spatial scale
                   size_t,              // spatial_bins_x
                   size_t,              // spatial_bins_y
                   std::string          // mode
                   >
    PSROIPoolingSpecificParams;

typedef std::tuple<PSROIPoolingSpecificParams,
                   ov::element::Type,  // Model Type
                   std::string         // Device name
                   >
    PSROIPoolingLayerTestParams;

typedef std::tuple<PSROIPoolingLayerTestParams, CPUSpecificParams> PSROIPoolingLayerCPUTestParamsSet;

class PSROIPoolingLayerCPUTest : public testing::WithParamInterface<PSROIPoolingLayerCPUTestParamsSet>,
                                 virtual public ov::test::SubgraphBaseStaticTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PSROIPoolingLayerCPUTestParamsSet> obj) {
        std::vector<float> proposal;
        ov::Shape featureMapShape;
        size_t spatialBinsX;
        size_t spatialBinsY;
        float spatialScale;
        size_t groupSize;
        size_t outputDim;
        std::string mode;

        PSROIPoolingLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ov::element::Type netPr;
        PSROIPoolingSpecificParams psroiPar;
        std::tie(psroiPar, netPr, td) = basicParamsSet;
        std::tie(featureMapShape, proposal, outputDim, groupSize, spatialScale, spatialBinsX, spatialBinsY, mode) =
            psroiPar;
        std::ostringstream result;
        result << "PSROIPoolingTest_";
        result << std::to_string(obj.index) << "_";
        result << "binsX=" << spatialBinsX << "_";
        result << "binsY=" << spatialBinsY << "_";
        result << "spatialScale=" << spatialScale << "_";
        result << "outputD=" << outputDim << "_";
        result << "groupS=" << groupSize << "_";
        result << netPr.get_type_name() << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<float> proposal;
        ov::Shape featureMapShape;
        size_t spatialBinsX;
        size_t spatialBinsY;
        float spatialScale;
        size_t groupSize;
        size_t outputDim;
        std::string mode;
        PSROIPoolingLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        PSROIPoolingSpecificParams psroiPoolingParams;
        auto netPrecision = ov::element::dynamic;
        std::tie(psroiPoolingParams, netPrecision, targetDevice) = basicParamsSet;
        inType = outType = netPrecision;
        std::tie(featureMapShape, proposal, outputDim, groupSize, spatialScale, spatialBinsX, spatialBinsY, mode) =
            psroiPoolingParams;

        ov::Shape proposalShape = {proposal.size() / 5, 5};

        auto coords = std::make_shared<ov::op::v0::Constant>(ov::element::f32, proposalShape, proposal);
        ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(featureMapShape))};

        auto psroi = std::make_shared<ov::op::v0::PSROIPooling>(params[0],
                                                                coords,
                                                                outputDim,
                                                                groupSize,
                                                                spatialScale,
                                                                spatialBinsX,
                                                                spatialBinsY,
                                                                mode);
        psroi->get_rt_info() = getCPUInfo();
        selectedType = getPrimitiveType() + "_" + ov::element::Type(inType).get_type_name();

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(psroi)};
        function = std::make_shared<ov::Model>(results, params, "PSROIPooling");
    }
};

TEST_P(PSROIPoolingLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "PSROIPooling");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> resCPUParams{CPUSpecificParams{{nchw, nc}, {nchw}, {}, {}},
                                            CPUSpecificParams{{nhwc, nc}, {nhwc}, {}, {}},
                                            CPUSpecificParams{{nChw16c, nc}, {nChw16c}, {}, {}}};

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::bf16};

const std::vector<float> spatialScaleVector = {1.0f};

const std::vector<ov::Shape> inputShapeVector = {ov::Shape({2, 200, 20, 20}),
                                                 ov::Shape({2, 200, 20, 16}),
                                                 ov::Shape({2, 200, 16, 20}),
                                                 ov::Shape({3, 200, 16, 16})};

const std::vector<std::vector<float>> averagePropVector = {{0, 0.9f, 0.9f, 18.9f, 18.9f, 1, 0.9f, 0.9f, 18.9f, 18.9f},
                                                           {1, 1, 1, 15, 15}};

const std::vector<std::vector<float>> bilinearPropVector = {{0, 0.1f, 0.1f, 0.9f, 0.9f, 1, 0.1f, 0.1f, 0.9f, 0.9f},
                                                            {1, 0.1f, 0.1f, 0.9f, 0.9f}};

const auto psroiPoolingAverageParams = ::testing::Combine(::testing::ValuesIn(inputShapeVector),
                                                          ::testing::ValuesIn(averagePropVector),
                                                          ::testing::Values(50),
                                                          ::testing::Values(2),
                                                          ::testing::ValuesIn(spatialScaleVector),
                                                          ::testing::Values(1),
                                                          ::testing::Values(1),
                                                          ::testing::Values("average"));

const auto psroiPoolingBilinearParams = ::testing::Combine(::testing::Values(ov::Shape({3, 32, 20, 20})),
                                                           ::testing::ValuesIn(bilinearPropVector),
                                                           ::testing::Values(4),
                                                           ::testing::Values(3),
                                                           ::testing::ValuesIn(spatialScaleVector),
                                                           ::testing::Values(4),
                                                           ::testing::Values(2),
                                                           ::testing::Values("bilinear"));

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest,
                         PSROIPoolingLayerCPUTest,
                         ::testing::Combine(::testing::Combine(psroiPoolingAverageParams,
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(resCPUParams))),
                         PSROIPoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingBilinearLayoutTest,
                         PSROIPoolingLayerCPUTest,
                         ::testing::Combine(::testing::Combine(psroiPoolingBilinearParams,
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(resCPUParams))),
                         PSROIPoolingLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
