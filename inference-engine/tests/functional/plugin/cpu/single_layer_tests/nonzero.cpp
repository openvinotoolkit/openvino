// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>  // input shape
> NonZeroSpecificParams;

typedef std::tuple<
        NonZeroSpecificParams,
        InferenceEngine::Precision,     // Net precision
        LayerTestsUtils::TargetDevice   // Device name
> NonZeroLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::NonZeroLayerTestParams,
        CPUSpecificParams> NonZeroLayerCPUTestParamsSet;

class NonZeroLayerCPUTest : public testing::WithParamInterface<NonZeroLayerCPUTestParamsSet>,
                          virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NonZeroLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::NonZeroLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        Precision netPrc = Precision::FP32;
        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;

        NonZeroSpecificParams rangePar;
        std::tie(rangePar, netPrc, td) = basicParamsSet;
        std::tie(shapes) = rangePar;

        std::ostringstream result;
        result << "NonZeroTest_" << std::to_string(obj.index) << "_";
        result << "NetPr_" << netPrc.name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CommonTestUtils::vec2str(shapes.second[0]) << "_";
        return result.str();
    }
protected:
void SetUp() override {
    CPULayerTestsDefinitions::NonZeroLayerTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    std::tie(basicParamsSet, cpuParams) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    CPULayerTestsDefinitions::NonZeroSpecificParams nonZeroParams;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
    std::tie(nonZeroParams, netPrecision, targetDevice) = basicParamsSet;

    std::tie(shapes) = nonZeroParams;
    InferenceEngine::Precision netPrc = Precision::FP32;
    inPrc = netPrc;
    outPrc = Precision::I32;
    targetStaticShapes = shapes.second;
    inputDynamicShapes = shapes.first;
    auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc);
    const auto& paramNode = ngraph::builder::makeParams(ngNetPrc, {targetStaticShapes.front().front()});

    auto nonZero = std::make_shared<ngraph::opset3::NonZero>(paramNode[0]);
    nonZero->get_rt_info() = getCPUInfo();
    selectedType = std::string("ref_") + inPrc.name();
    paramNode[0]->set_friendly_name("input");
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nonZero)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{paramNode}, "NonZero");
    functionRefs = ngraph::clone_function(*function);
}
};

TEST_P(NonZeroLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "NonZero");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    return std::vector<CPUSpecificParams> {CPUSpecificParams{{}, {nc}, {}, {}}};;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U32,
        InferenceEngine::Precision::U8
};

std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic = {
        {{ngraph::PartialShape {-1}},
         {{{1000}}, {{2000}}, {{3000}}}},
        {{ngraph::PartialShape {-1, -1}},
         {{{4, 1000}}, {{4, 2000}}, {{4, 3000}}}},
        {{ngraph::PartialShape {-1, -1, -1}},
         {{{4, 4, 1000}}, {{4, 4, 2000}}, {{4, 4, 3000}}}},
        {{ngraph::PartialShape {-1, -1, -1, -1}},
         {{{4, 4, 4, 1000}}, {{4, 4, 4, 2000}}, {{4, 4, 4, 3000}}}},
        {{ngraph::PartialShape {-1, -1, -1, -1, -1}},
         {{{4, 4, 4, 2, 1000}}, {{4, 4, 4, 2, 2000}}, {{4, 4, 4, 2, 3000}}}}
};
std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesStatic = {
        {{}, {{{1000}}}},
        {{}, {{{4, 1000}}}},
        {{}, {{{4, 2, 1000}}}},
        {{}, {{{4, 4, 2, 1000}}}},
        {{}, {{{4, 4, 4, 2, 1000}}}}
};

const auto nonZeroParDynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic)
);
const auto nonZeroParStatic = ::testing::Combine(
        ::testing::ValuesIn(inShapesStatic)
);
const auto params3dDynamic = ::testing::Combine(
        ::testing::Combine(
                nonZeroParDynamic,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params3dStatic = ::testing::Combine(
        ::testing::Combine(
                nonZeroParStatic,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_NonZeroStaticLayoutTest, NonZeroLayerCPUTest,
                         params3dStatic, NonZeroLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonZeroDynamicLayoutTest, NonZeroLayerCPUTest,
                         params3dDynamic, NonZeroLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
