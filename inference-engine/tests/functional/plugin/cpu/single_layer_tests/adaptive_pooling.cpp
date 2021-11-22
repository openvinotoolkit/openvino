/// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<int> pooledSpatialShape;
    std::string mode;
    std::vector<InputShape> inputShape;
}  // namespace

using AdaPoolSpecificParams = std::tuple<
        std::vector<int>,        // pooled vector
        std::vector<InputShape>>;      // feature map shape

using AdaPoolLayerTestParams = std::tuple<
        AdaPoolSpecificParams,
        std::string,                        // mode
        ElementType,         // Net precision
        TargetDevice>;       // Device name

using AdaPoolLayerCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::AdaPoolLayerTestParams,
        CPUSpecificParams>;

class AdaPoolLayerCPUTest : public testing::WithParamInterface<AdaPoolLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AdaPoolLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        AdaPoolSpecificParams adaPar;
        std::tie(adaPar, mode, netPr, td) = basicParamsSet;
        std::tie(pooledSpatialShape, inputShape) = adaPar;
        std::ostringstream result;
        result << "AdaPoolTest_";
        result << "IS=(";
        for (const auto& shape : inputShape) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShape) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << "OS=" << CommonTestUtils::vec2str(pooledSpatialShape) << "(spat.)_";
        result << netPr << "_";
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
        ElementType netPrecision;
        std::tie(adaPoolParams, mode,  netPrecision, targetDevice) = basicParamsSet;
        std::tie(pooledSpatialShape, inputShape) = adaPoolParams;

        ngraph::Shape coordsShape = {pooledSpatialShape.size() };
        auto pooledParam = ngraph::builder::makeConstant<int32_t>(ngraph::element::i32, coordsShape, pooledSpatialShape);
        init_input_shapes(inputShape);
        auto params = ngraph::builder::makeDynamicParams(ngraph::element::f32, { inputDynamicShapes[0] });

        // we cannot create abstract Op to use polymorphism
        auto adapoolMax = std::make_shared<ngraph::opset8::AdaptiveMaxPool>(params[0], pooledParam, ngraph::element::i32);
        adapoolMax->get_rt_info() = getCPUInfo();
        auto adapoolAvg = std::make_shared<ngraph::opset8::AdaptiveAvgPool>(params[0], pooledParam);
        adapoolAvg->get_rt_info() = getCPUInfo();

        selectedType = std::string("unknown_FP32");
        if (netPrecision == ElementType::bf16) {
            rel_threshold = 1e-2;
        }
        function = (mode == "max" ? std::make_shared<ngraph::Function>(adapoolMax->outputs(), params, "AdaPoolMax") :
                    std::make_shared<ngraph::Function>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
    }
};

TEST_P(AdaPoolLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
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

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16
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

std::vector<std::vector<ov::Shape>> staticInput3DShapeVector = {{{1, 17, 3}, {3, 7, 5}}};

const std::vector<std::vector<InputShape>> input3DShapeVector = {
        {
                {{{-1, 17, -1}, {{1, 17, 3}, {3, 17, 5}, {3, 17, 5}}}},
                {{{{1, 10}, 20, {1, 10}}, {{1, 20, 5}, {2, 20, 4}, {3, 20, 6}}}}
        }
};

std::vector<std::vector<ov::Shape>> staticInput4DShapeVector = {{{1, 3, 1, 1}, {3, 17, 5, 2}}};

const std::vector<std::vector<InputShape>> input4DShapeVector = {
        {
                {{{-1, 3, -1, -1}, {{1, 3, 1, 1}, {3, 3, 5, 2}, {3, 3, 5, 2}}}},
                {{{{1, 10}, 3, {1, 10}, {1, 10}}, {{2, 3, 10, 6}, {3, 3, 6, 5}, {3, 3, 6, 5}}}}
        }
};

std::vector<std::vector<ov::Shape>> staticInput5DShapeVector = {{{ 1, 17, 2, 5, 2}, {3, 17, 4, 5, 4}}};

const std::vector<std::vector<InputShape>> input5DShapeVector = {
        {
                {{{-1, 17, -1, -1, -1}, {{1, 17, 2, 5, 2}, {3, 17, 4, 5, 4}, {3, 17, 4, 5, 4}}}},
                {{{{1, 10}, 3, {1, 10}, {1, 10}, {1, 10}}, {{3, 3, 2, 5, 2}, {1, 3, 4, 5, 4}, {1, 3, 4, 5, 4}}}}
        }
};

const auto adaPool3DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(input3DShapeVector)      // feature map shape
);

const auto adaPool4DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled4DVector),         // output spatial shape
        ::testing::ValuesIn(input4DShapeVector)     // feature map shape
);

const auto adaPool5DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled5DVector),         // output spatial shape
        ::testing::ValuesIn(input5DShapeVector)     // feature map shape
);

const auto staticAdaPool3DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput3DShapeVector))      // feature map shape
);

const auto staticAdaPool4DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled4DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput4DShapeVector))     // feature map shape
);

const auto staticAdaPool5DParams = ::testing::Combine(
        ::testing::ValuesIn(pooled5DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput5DShapeVector))     // feature map shape
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

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolAvg3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool3DParams,
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("3D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolAvg4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool4DParams,
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("4D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolAvg5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool5DParams,
                                         ::testing::Values("avg"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("5D", "avg"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolMax3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool3DParams,
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("3D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolMax4DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool4DParams,
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("4D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticAdaPoolMax5DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAdaPool5DParams,
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("5D", "max"))),
                         AdaPoolLayerCPUTest::getTestCaseName);


// in 1-channel cases  {..., 1, 1, 1} shape cannot be correctly resolved on oneDnn level, so it was removed from instances

const std::vector<std::vector<InputShape>> input3DShape1Channel = {
     {{{-1, -1, -1},
        {{1, 1, 2}, {2, 1, 2}, {2, 1, 2}}}}
};

const std::vector<std::vector<InputShape>> input4DShape1Channel = {
     {{{-1, -1, -1, -1},
        {{1, 1, 1, 2}, {2, 1, 2, 1}}}}
};

const std::vector<std::vector<InputShape>> input5DShape1Channel = {
     {{{-1, -1, -1, -1, -1},
        {{1, 1, 1, 1, 2}, {2, 1, 1, 2, 1}}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_AdaPool_1ch_Avg3DLayoutTest, AdaPoolLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::Combine(
                                                 ::testing::ValuesIn(std::vector<std::vector<int>> {
                                                         {1}, {2}}),
                                                 ::testing::ValuesIn(input3DShape1Channel)),
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
                                                 ::testing::ValuesIn(input4DShape1Channel)),
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
                                                 ::testing::ValuesIn(input5DShape1Channel)),
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
                                                 ::testing::ValuesIn(input3DShape1Channel)),
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
                                                 ::testing::ValuesIn(input4DShape1Channel)),
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
                                                 ::testing::ValuesIn(input5DShape1Channel)),
                                         ::testing::Values("max"),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}})),
                         AdaPoolLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
