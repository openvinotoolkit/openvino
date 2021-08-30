/// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<int> pooledSpatialShape;
    std::string mode;
    std::vector<size_t> inputShape;
}  // namespace

typedef std::tuple<
        std::vector<int>,        // pooled vector
        std::vector<size_t>      // feature map shape
> AdaPoolSpecificParams;

typedef std::tuple<
        AdaPoolSpecificParams,
        std::string,                        // mode
        InferenceEngine::Precision,         // Net precision
        LayerTestsUtils::TargetDevice       // Device name
> AdaPoolLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::AdaPoolLayerTestParams,
        CPUSpecificParams> AdaPoolLayerCPUTestParamsSet;

class AdaPoolLayerCPUTest : public testing::WithParamInterface<AdaPoolLayerCPUTestParamsSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AdaPoolLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        Precision netPr;
        AdaPoolSpecificParams adaPar;
        std::tie(adaPar, mode, netPr, td) = basicParamsSet;
        std::tie(pooledSpatialShape, inputShape) = adaPar;
        std::ostringstream result;
        result << "AdaPoolTest_";
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "OS=" << CommonTestUtils::vec2str(pooledSpatialShape) << "(spat.)_";
        result << netPr.name() << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::AdaPoolSpecificParams adaPoolParams;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(adaPoolParams, mode,  netPrecision, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        std::tie(pooledSpatialShape, inputShape) = adaPoolParams;

        ngraph::Shape coordsShape = {pooledSpatialShape.size() };
        auto pooledParam = ngraph::builder::makeConstant<int32_t>(ngraph::element::i32, coordsShape, pooledSpatialShape);
        auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});

        // we cannot create abstract Op to use polymorphism
        auto adapoolMax = std::make_shared<ngraph::opset8::AdaptiveMaxPool>(params[0], pooledParam, ngraph::element::i32);
        adapoolMax->get_rt_info() = getCPUInfo();
        auto adapoolAvg = std::make_shared<ngraph::opset8::AdaptiveAvgPool>(params[0], pooledParam);
        adapoolAvg->get_rt_info() = getCPUInfo();

        selectedType = std::string("unknown_FP32");
        threshold = 1e-2;
        function = (mode == "max" ? std::make_shared<ngraph::Function>(adapoolMax->outputs(), params, "AdaPoolMax") :
                    std::make_shared<ngraph::Function>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
    }
};

TEST_P(AdaPoolLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "AdaptivePooling");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::string dims = "3D", std::string modeStr = "max") {
    std::vector<CPUSpecificParams> resCPUParams;
    if (modeStr == "max") {
        if (dims == "5D") {
            resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});  // i.e. two equal output layouts
            resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc, ncdhw}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x}, {nCdhw16c, ncdhw}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x}, {nCdhw8c, ncdhw}, {}, {}});
            }
        } else if (dims == "4D") {
            resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}});  // i.e. two equal output layouts
            resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nchw}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c, nchw}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nchw}, {}, {}});
            }
        } else {
            resCPUParams.push_back(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}});  // i.e. two equal output layouts
            resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc, ncw}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw16c, x}, {nCw16c, ncw}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw8c, x}, {nCw8c, ncw}, {}, {}});
            }
        }
    } else {
        if (dims == "5D") {
            resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});
            resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x}, {nCdhw16c}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x}, {nCdhw8c}, {}, {}});
            }
        } else if (dims == "4D") {
            resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}});
            resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c}, {}, {}});
            }
        } else {
            resCPUParams.push_back(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}});
            resCPUParams.push_back(CPUSpecificParams{{nwc, x}, {nwc}, {}, {}});
            if (with_cpu_x86_avx512f()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw16c, x}, {nCw16c}, {}, {}});
            } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
                resCPUParams.push_back(CPUSpecificParams{{nCw8c, x}, {nCw8c}, {}, {}});
            }
        }
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16
};

const std::vector<std::vector<int>> pooled3DVector = {
        { 1 },
        { 3 },
        { 5 }
};
const std::vector<std::vector<int>> pooled4DVector = {
        { 1, 1 },
        { 3, 5 },
        { 5, 5 }
};

const std::vector<std::vector<int>> pooled5DVector = {
        { 1, 1, 1 },
        { 3, 5, 1 },
        { 3, 5, 3 },
};

const std::vector<std::vector<size_t>> input3DShapeVector = {
        SizeVector({ 1, 17, 3 }),
        SizeVector({ 3, 17, 5 }),
};

const std::vector<std::vector<size_t>> input4DShapeVector = {
        SizeVector({ 1, 3, 1, 1 }),
        SizeVector({ 3, 17, 5, 2 }),
};

const std::vector<std::vector<size_t>> input5DShapeVector = {
        SizeVector({ 1, 17, 2, 5, 2 }),
        SizeVector({ 3, 17, 4, 5, 4 }),
};

const auto adaPool3DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(input3DShapeVector)     // feature map shape
);

const auto adaPool4DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled4DVector),         // output spatial shape
        ::testing::ValuesIn(input4DShapeVector)     // feature map shape
);

const auto adaPool5DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled5DVector),         // output spatial shape
        ::testing::ValuesIn(input5DShapeVector)     // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool3DParams,
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("3D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool4DParams,
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("4D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolAvg5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool5DParams,
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("5D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool3DParams,
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("3D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool4DParams,
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("4D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPoolMax5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         adaPool5DParams,
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("5D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

// in 1-channel cases  {..., 1, 1, 1} shape cannot be correctly resolved on oneDnn level, so it was removed from instances

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Avg3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1}, {2}}),
                                                 ::testing::ValuesIn(std::vector<std::vector<size_t>> {
                                                         SizeVector{1, 1, 2}, SizeVector{2, 1, 2}})),
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Avg4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1, 1},
                                                         {2, 2}
                                                 }),
                                                 ::testing::ValuesIn(std::vector<std::vector<size_t>> {
                                                         SizeVector{1, 1, 1, 2},
                                                         SizeVector{2, 1, 2, 1}
                                                 })),
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Avg5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1, 1, 1}, {2, 2, 2}}),
                                                 ::testing::ValuesIn(std::vector<std::vector<size_t>> {
                                                         SizeVector{1, 1, 1, 1, 2}, SizeVector{2, 1, 1, 2, 1}})),
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Max3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1}, {2}}),
                                                 ::testing::ValuesIn(std::vector<std::vector<size_t>> {
                                                         SizeVector{1, 1, 2}, SizeVector{2, 1, 2}})),
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{ncw, x}, {ncw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Max4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1, 1}, {2, 2}}),
                                                 ::testing::ValuesIn(std::vector<std::vector<size_t>> {
                                                         SizeVector{1, 1, 1, 2}, SizeVector{2, 1, 2, 1}})),
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{nchw, x}, {nchw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Max5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1, 1, 1},
                                                         {2, 2, 2}
                                                 }),
                                                 ::testing::ValuesIn(std::vector<std::vector<size_t>> {
                                                         SizeVector{1, 1, 1, 1, 2},
                                                         SizeVector{2, 1, 1, 2, 1}
                                                 })),
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
