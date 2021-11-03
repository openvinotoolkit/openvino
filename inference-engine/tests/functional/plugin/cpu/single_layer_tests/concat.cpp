// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using ConcatInputShapes = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

typedef std::tuple<
        size_t,             // Concat axis
        ConcatInputShapes,  // Input shapes
        ElementType,        // Network precision
        CPUSpecificParams
> concatCPUTestParams;

class ConcatLayerCPUTest : public testing::WithParamInterface<concatCPUTestParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatCPUTestParams> obj) {
        int axis;
        ConcatInputShapes inputShapes;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(axis, inputShapes, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        if (!inputShapes.first.empty()) {
            result << "IS=(";
            result << CommonTestUtils::partialShape2str(inputShapes.first) << ")_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << "axis=" << axis << "_";
        result << "netPRC=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        int axis;
        ConcatInputShapes inputShape;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(axis, inputShape, netPrecision, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + InferenceEngine::details::convertPrecision(netPrecision).name();

        if (!inputShape.first.empty()) {
            inputDynamicShapes = inputShape.first;
        } else {
            const auto &inShapes = inputShape.second.front();
            for (const auto &shape : inShapes) {
                inputDynamicShapes.push_back(shape);
            }
        }
        targetStaticShapes = inputShape.second;

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);

        concat->get_rt_info() = getCPUInfo();

        function = std::make_shared<ngraph::Function>(concat, params, "ConcatCPU");
        // function = makeNgraphFunction(netPrecision, params, concat, "ConcatCPU");
    }
};

TEST_P(ConcatLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
//     CheckPluginRelatedresult(executableNetwork, "Concatenation");
}

namespace {
const auto planar_4D_ref = CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref"};
const auto planar_5D_ref = CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref"}, "ref"};

const auto planar_4D = CPUSpecificParams{{nchw}, {nchw}, {}, "unknown"};
const auto planar_5D = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "unknown"};

const auto planarChannels_4D = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto planarChannels_5D = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};

const auto blocked8_4D = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "unknown"};
const auto blocked8_5D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "unknown"};

const auto blocked8_4D_ref = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto blocked8_5D_ref = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto blocked16_4D = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "unknown"};
const auto blocked16_5D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "unknown"};

const auto blocked16_4D_ref = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto blocked16_5D_ref = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

// List of precisions natively supported by mkldnn.
const std::vector<ElementType> netPrecisions = {
        ElementType::i8,
        ElementType::i32,
        ElementType::f32,
        ElementType::bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block8_static, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1, 2, 3),
                                ::testing::Values(ConcatInputShapes{ {}, {{{2, 16, 3, 5}, {2, 16, 3, 5}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D_ref, planarChannels_4D, blocked8_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16_static, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1, 2, 3),
                                ::testing::Values(ConcatInputShapes{ {}, {{{3, 32, 3, 5}, {3, 32, 3, 5}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes4D_Block_axis1 = {
        {
            // dynamic
            {{-1, 32, -1, -1}, {-1, 16, -1, -1}, {-1, 64, -1, -1}},
            // target
            {
                {{2, 32, 5, 7}, {2, 16, 5, 7}, {2, 64, 5, 7}},
                {{1, 32, 10, 2}, {1, 16, 10, 2}, {1, 64, 10, 2}},
                {{3, 32, 1, 8}, {3, 16, 1, 8}, {3, 64, 1, 8}},
            }
        },
        {
            // dynamic
            {{{1, 5}, 32, {1, 10}, {2, 8}}, {{1, 3}, 16, {1, 10}, {2, 8}}, {{1, 3}, 64, {1, 10}, {2, 8}}},
            // target
            {
                {{2, 32, 5, 7}, {2, 16, 5, 7}, {2, 64, 5, 7}},
                {{1, 32, 10, 2}, {1, 16, 10, 2}, {1, 64, 10, 2}},
                {{3, 32, 1, 8}, {3, 16, 1, 8}, {3, 64, 1, 8}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block_dynamic_axis_1, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::ValuesIn(inputShapes4D_Block_axis1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_4D_ref, blocked16_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes4D_axis1 = {
        {
            // dynamic
            {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
            // target
            {
                {{2, 32, 5, 7}, {2, 16, 5, 7}, {2, 64, 5, 7}},
                {{1, 18, 10, 2}, {1, 5, 10, 2}, {1, 45, 10, 2}},
                {{3, 8, 1, 8}, {3, 3, 1, 8}, {3, 1, 1, 8}},
            }
        },
        {
            // dynamic
            {{{1, 3}, {8, 32}, {1, 10}, {2, 8}}, {{1, 3}, {3, 16}, {1, 10}, {2, 8}}, {{1, 3}, {1, 64}, {1, 10}, {2, 8}}},
            // target
            {
                {{2, 32, 5, 7}, {2, 16, 5, 7}, {2, 64, 5, 7}},
                {{1, 18, 10, 2}, {1, 5, 10, 2}, {1, 45, 10, 2}},
                {{3, 8, 1, 8}, {3, 3, 1, 8}, {3, 1, 1, 8}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_dynamic_axis_1, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::ValuesIn(inputShapes4D_axis1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D_ref, planarChannels_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes4D_Block_axis2 = {
        {
            // dynamic
            {{-1, 16, -1, -1}, {-1, 16, -1, -1}, {-1, 16, -1, -1}},
            // target
            {
                {{2, 16, 5, 7}, {2, 16, 1, 7}, {2, 16, 10, 7}},
                {{1, 16, 16, 2}, {1, 16, 3, 2}, {1, 16, 5, 2}},
                {{3, 16, 2, 8}, {3, 16, 11, 8}, {3, 16, 1, 8}},
            }
        },
        {
            // dynamic
            {{{1, 3}, 16, {2, 16}, {2, 8}}, {{1, 3}, 16, {1, 11}, {2, 8}}, {{1, 3}, 16, {1, 10}, {2, 8}}},
            // target
            {
                {{2, 16, 5, 7}, {2, 16, 1, 7}, {2, 16, 10, 7}},
                {{1, 16, 16, 2}, {1, 16, 3, 2}, {1, 16, 5, 2}},
                {{3, 16, 2, 8}, {3, 16, 11, 8}, {3, 16, 1, 8}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block_dynamic_axis_2, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::ValuesIn(inputShapes4D_Block_axis2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_4D_ref, blocked16_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes4D_axis2 = {
        {
            // dynamic
            {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
            // target
            {
                {{2, 16, 5, 7}, {2, 16, 1, 7}, {2, 16, 10, 7}},
                {{1, 16, 16, 2}, {1, 16, 3, 2}, {1, 16, 5, 2}},
                {{3, 16, 2, 8}, {3, 16, 11, 8}, {3, 16, 1, 8}},
            }
        },
        {
            // dynamic
            {{{1, 3}, {1, 16}, {2, 16}, {2, 8}}, {{1, 3}, {1, 16}, {1, 11}, {2, 8}}, {{1, 3}, {1, 16}, {1, 10}, {2, 8}}},
            // target
            {
                {{2, 16, 5, 7}, {2, 16, 1, 7}, {2, 16, 10, 7}},
                {{1, 16, 16, 2}, {1, 16, 3, 2}, {1, 16, 5, 2}},
                {{3, 16, 2, 8}, {3, 16, 11, 8}, {3, 16, 1, 8}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_dynamic_axis_2, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::ValuesIn(inputShapes4D_axis2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D_ref, planarChannels_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes4D_Block_axis3 = {
        {
            // dynamic
            {{-1, 32, -1, -1}, {-1, 32, -1, -1}, {-1, 32, -1, -1}},
            // target
            {
                {{2, 32, 4, 5}, {2, 32, 4, 1}, {2, 32, 4, 10}},
                {{1, 32, 1, 16}, {1, 32, 1, 3}, {1, 32, 1, 5}},
                {{3, 32, 7, 2}, {3, 32, 7, 11}, {3, 32, 7, 1}},
            }
        },
        {
            // dynamic
            {{{1, 3}, 32, {1, 7}, {2, 16}}, {{1, 3}, 32, {1, 7}, {1, 11}}, {{1, 3}, 32, {1, 7}, {1, 10}}},
            // target
            {
                {{2, 32, 4, 5}, {2, 32, 4, 1}, {2, 32, 4, 10}},
                {{1, 32, 1, 16}, {1, 32, 1, 3}, {1, 32, 1, 5}},
                {{3, 32, 7, 2}, {3, 32, 7, 11}, {3, 32, 7, 1}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block_dynamic_axis_3, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::ValuesIn(inputShapes4D_Block_axis3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_4D_ref, blocked16_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes4D_axis3 = {
        {
            // dynamic
            {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
            // target
            {
                {{2, 32, 4, 5}, {2, 32, 4, 1}, {2, 32, 4, 10}},
                {{1, 32, 1, 16}, {1, 32, 1, 3}, {1, 32, 1, 5}},
                {{3, 32, 7, 2}, {3, 32, 7, 11}, {3, 32, 7, 1}},
            }
        },
        {
            // dynamic
            {{{1, 3}, {1, 32}, {1, 7}, {2, 16}}, {{1, 3}, {1, 32}, {1, 7}, {1, 11}}, {{1, 3}, {1, 32}, {1, 7}, {1, 10}}},
            // target
            {
                {{2, 32, 4, 5}, {2, 32, 4, 1}, {2, 32, 4, 10}},
                {{1, 32, 1, 16}, {1, 32, 1, 3}, {1, 32, 1, 5}},
                {{3, 32, 7, 2}, {3, 32, 7, 11}, {3, 32, 7, 1}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_dynamic_axis_3, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::ValuesIn(inputShapes4D_axis3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D_ref, planarChannels_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block8_static, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(ConcatInputShapes{ {}, {{{2, 16, 3, 5, 7}, {2, 16, 3, 5, 7}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D_ref, planarChannels_5D, blocked8_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block16_static, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(ConcatInputShapes{ {}, {{{2, 32, 3, 5, 7}, {2, 32, 3, 5, 7}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_Block_axis1 = {
        {
            // dynamic
            {{-1, 32, -1, -1, -1}, {-1, 16, -1, -1, -1}, {-1, 64, -1, -1, -1}},
            // target
            {
                {{2, 32, 5, 7, 6}, {2, 16, 5, 7, 6}, {2, 64, 5, 7, 6}},
                {{1, 32, 10, 2, 8}, {1, 16, 10, 2, 8}, {1, 64, 10, 2, 8}},
                {{3, 32, 1, 8, 10}, {3, 16, 1, 8, 10}, {3, 64, 1, 8, 10}},
            }
        },
        {
            // dynamic
            {{{1, 3}, 32, {1, 10}, {2, 8}, {6, 10}}, {{1, 3}, 16, {1, 10}, {2, 8}, {6, 10}}, {{1, 3}, 64, {1, 10}, {2, 8}, {6, 10}}},
            // target
            {
                {{2, 32, 5, 7, 6}, {2, 16, 5, 7, 6}, {2, 64, 5, 7, 6}},
                {{1, 32, 10, 2, 8}, {1, 16, 10, 2, 8}, {1, 64, 10, 2, 8}},
                {{3, 32, 1, 8, 10}, {3, 16, 1, 8, 10}, {3, 64, 1, 8, 10}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_1, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::ValuesIn(inputShapes5D_Block_axis1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_axis1 = {
        {
            // dynamic
            {{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}},
            // target
            {
                {{2, 5, 5, 7, 6}, {2, 16, 5, 7, 6}, {2, 1, 5, 7, 6}},
                {{1, 3, 10, 2, 8}, {1, 20, 10, 2, 8}, {1, 17, 10, 2, 8}},
                {{3, 4, 1, 8, 10}, {3, 5, 1, 8, 10}, {3, 5, 1, 8, 10}},
            }
        },
        {
            // dynamic
            {{{1, 3}, {3, 5}, {1, 10}, {2, 8}, {6, 10}}, {{1, 3}, {5, 20}, {1, 10}, {2, 8}, {4, 10}}, {{1, 3}, {1, 17}, {1, 10}, {2, 8}, {6, 10}}},
            // target
            {
                {{2, 5, 5, 7, 6}, {2, 16, 5, 7, 6}, {2, 1, 5, 7, 6}},
                {{1, 3, 10, 2, 8}, {1, 20, 10, 2, 8}, {1, 17, 10, 2, 8}},
                {{3, 4, 1, 8, 10}, {3, 5, 1, 8, 10}, {3, 5, 1, 8, 10}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_1, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::ValuesIn(inputShapes5D_axis1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D_ref, planarChannels_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_Block_axis2 = {
        {
            // dynamic
            {{-1, 16, -1, -1, -1}, {-1, 16, -1, -1, -1}, {-1, 16, -1, -1, -1}},
            // target
            {
                {{2, 16, 5, 8, 7}, {2, 16, 1, 8, 7}, {2, 16, 10, 8, 7}},
                {{1, 16, 16, 1, 2}, {1, 16, 3, 1, 2}, {1, 16, 5, 1, 2}},
                {{3, 16, 2, 5, 8}, {3, 16, 11, 5, 8}, {3, 16, 1, 5, 8}},
            }
        },
        {
            // dynamic
            {{{1, 3}, 16, {2, 16}, {1, 8}, {2, 8}}, {{1, 5}, 16, {1, 11}, {1, 8}, {1, 8}}, {{1, 6}, 16, {1, 10}, {1, 8}, {2, 10}}},
            // target
            {
                {{2, 16, 5, 8, 7}, {2, 16, 1, 8, 7}, {2, 16, 10, 8, 7}},
                {{1, 16, 16, 1, 2}, {1, 16, 3, 1, 2}, {1, 16, 5, 1, 2}},
                {{3, 16, 2, 5, 8}, {3, 16, 11, 5, 8}, {3, 16, 1, 5, 8}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_2, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::ValuesIn(inputShapes5D_Block_axis2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_axis2 = {
        {
            // dynamic
            {{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}},
            // target
            {
                {{2, 4, 5, 8, 7}, {2, 4, 1, 8, 7}, {2, 4, 10, 8, 7}},
                {{1, 20, 16, 1, 2}, {1, 20, 3, 1, 2}, {1, 20, 5, 1, 2}},
                {{3, 8, 2, 5, 8}, {3, 8, 11, 5, 8}, {3, 8, 1, 5, 8}},
            }
        },
        {
            // dynamic
            {{{1, 3}, {4, 20}, {1, 16}, {1, 8}, {2, 8}}, {{1, 3}, {4, 20}, {1, 11}, {1, 10}, {1, 15}}, {{1, 3}, {1, 20}, {1, 15}, {1, 10}, {2, 8}}},
            // target
            {
                {{2, 4, 5, 8, 7}, {2, 4, 1, 8, 7}, {2, 4, 10, 8, 7}},
                {{1, 20, 16, 1, 2}, {1, 20, 3, 1, 2}, {1, 20, 5, 1, 2}},
                {{3, 8, 2, 5, 8}, {3, 8, 11, 5, 8}, {3, 8, 1, 5, 8}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_2, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::ValuesIn(inputShapes5D_axis2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D_ref, planarChannels_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_Block_axis3 = {
        {
            // dynamic
            {{-1, 32, -1, -1, -1}, {-1, 32, -1, -1, -1}, {-1, 32, -1, -1, -1}},
            // target
            {
                {{2, 32, 4, 5, 7}, {2, 32, 4, 1, 7}, {2, 32, 4, 10, 7}},
                {{1, 32, 1, 16, 3}, {1, 32, 1, 3, 3}, {1, 32, 1, 5, 3}},
                {{3, 32, 7, 2, 4}, {3, 32, 7, 11, 4}, {3, 32, 7, 1, 4}},
            }
        },
        {
            // dynamic
            {{{1, 3}, 32, {1, 7}, {2, 16}, {3, 7}}, {{1, 5}, 32, {1, 7}, {1, 11}, {3, 7}}, {{1, 6}, 32, {1, 15}, {1, 10}, {1, 20}}},
            // target
            {
                {{2, 32, 4, 5, 7}, {2, 32, 4, 1, 7}, {2, 32, 4, 10, 7}},
                {{1, 32, 1, 16, 3}, {1, 32, 1, 3, 3}, {1, 32, 1, 5, 3}},
                {{3, 32, 7, 2, 4}, {3, 32, 7, 11, 4}, {3, 32, 7, 1, 4}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_3, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::ValuesIn(inputShapes5D_Block_axis3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_axis3 = {
        {
            // dynamic
            {{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}},
            // target
            {
                {{2, 32, 4, 5, 7}, {2, 32, 4, 1, 7}, {2, 32, 4, 10, 7}},
                {{1, 11, 1, 16, 3}, {1, 11, 1, 3, 3}, {1, 11, 1, 5, 3}},
                {{3, 7, 7, 2, 4}, {3, 7, 7, 11, 4}, {3, 7, 7, 1, 4}},
            }
        },
        {
            // dynamic
            {{{1, 7}, {7, 32}, {1, 7}, {1, 16}, {3, 14}}, {{1, 7}, {7, 32}, {1, 10}, {1, 11}, {3, 7}}, {{1, 7}, {1, 32}, {1, 10}, {1, 10}, {1, 10}}},
            // target
            {
                {{2, 32, 4, 5, 7}, {2, 32, 4, 1, 7}, {2, 32, 4, 10, 7}},
                {{1, 11, 1, 16, 3}, {1, 11, 1, 3, 3}, {1, 11, 1, 5, 3}},
                {{3, 7, 7, 2, 4}, {3, 7, 7, 11, 4}, {3, 7, 7, 1, 4}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_3, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::ValuesIn(inputShapes5D_axis3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D_ref, planarChannels_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_Block_axis4 = {
        {
            // dynamic
            {{-1, 32, -1, -1, -1}, {-1, 32, -1, -1, -1}, {-1, 32, -1, -1, -1}},
            // target
            {
                {{2, 32, 4, 5, 5}, {2, 32, 4, 5, 1}, {2, 32, 4, 5, 10}},
                {{1, 32, 1, 1, 16}, {1, 32, 1, 1, 3}, {1, 32, 1, 1, 5}},
                {{3, 32, 7, 9, 2}, {3, 32, 7, 9, 11}, {3, 32, 7, 9, 1}},
            }
        },
        {
            // dynamic
            {{{1, 15}, 32, {1, 10}, {1, 10}, {1, 16}}, {{1, 15}, 32, {1, 10}, {1, 10}, {1, 11}}, {{1, 15}, 32, {1, 10}, {1, 10}, {1, 11}}},
            // target
            {
                {{2, 32, 4, 5, 5}, {2, 32, 4, 5, 1}, {2, 32, 4, 5, 10}},
                {{1, 32, 1, 1, 16}, {1, 32, 1, 1, 3}, {1, 32, 1, 1, 5}},
                {{3, 32, 7, 9, 2}, {3, 32, 7, 9, 11}, {3, 32, 7, 9, 1}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_4, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::ValuesIn(inputShapes5D_Block_axis4),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes5D_axis4 = {
        {
            // dynamic
            {{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}},
            // target
            {
                {{2, 1, 4, 5, 5}, {2, 1, 4, 5, 1}, {2, 1, 4, 5, 10}},
                {{1, 4, 1, 1, 16}, {1, 4, 1, 1, 3}, {1, 4, 1, 1, 5}},
                {{3, 14, 7, 9, 2}, {3, 14, 7, 9, 11}, {3, 14, 7, 9, 1}},
            }
        },
        {
            // dynamic
            {{{1, 3}, {1, 14}, {1, 7}, {1, 10}, {2, 16}}, {{1, 3}, {1, 14}, {1, 7}, {1, 9}, {1, 11}}, {{1, 3}, {1, 14}, {1, 7}, {1, 9}, {1, 10}}},
            // target
            {
                {{2, 1, 4, 5, 5}, {2, 1, 4, 5, 1}, {2, 1, 4, 5, 10}},
                {{1, 4, 1, 1, 16}, {1, 4, 1, 1, 3}, {1, 4, 1, 1, 5}},
                {{3, 14, 7, 9, 2}, {3, 14, 7, 9, 11}, {3, 14, 7, 9, 1}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_4, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::ValuesIn(inputShapes5D_axis4),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D_ref, planarChannels_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes_byBatch = {
        { {}, {{{5, 2, 2, 2}, {2, 2, 2, 2}}} },
        { {}, {{{1, 3, 5}, {3, 3, 5}}} },
        { {}, {{{4, 3, 2}, {1, 3, 2}}} },
        // 5D
        {
            // dynamic
            {{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}},
            // target
            {
                {{10, 32, 4, 5, 5}, {5, 32, 4, 5, 5}, {1, 32, 4, 5, 5}},
                {{4, 7, 1, 1, 3}, {7, 7, 1, 1, 3}, {1, 7, 1, 1, 3}},
                {{3, 20, 7, 9, 1}, {3, 20, 7, 9, 1}, {6, 20, 7, 9, 1}},
            }
        },
        {
            // dynamic
            {{{3, 10}, {7, 32}, {1, 9}, {1, 10}, {1, 5}}, {{3, 7}, {7, 32}, {1, 7}, {1, 9}, {1, 5}}, {{1, 6}, {7, 32}, {1, 7}, {1, 9}, {1, 5}}},
            // target
            {
                {{10, 32, 4, 5, 5}, {5, 32, 4, 5, 5}, {1, 32, 4, 5, 5}},
                {{4, 7, 1, 1, 3}, {7, 7, 1, 1, 3}, {1, 7, 1, 1, 3}},
                {{3, 20, 7, 9, 1}, {3, 20, 7, 9, 1}, {6, 20, 7, 9, 1}},
            }
        },
        // 4D
        {
            // dynamic
            {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
            // target
            {
                {{10, 32, 4, 5}, {5, 32, 4, 5}, {1, 32, 4, 5}},
                {{4, 7, 1, 1}, {7, 7, 1, 1}, {1, 7, 1, 1}},
                {{3, 20, 7, 9}, {3, 20, 7, 9}, {6, 20, 7, 9}},
            }
        },
        {
            // dynamic
            {{{1, 10}, {1, 32}, {1, 7}, {1, 9}}, {{3, 7}, {7, 32}, {1, 7}, {1, 9}}, {{1, 6}, {7, 32}, {1, 7}, {1, 9}}},
            // target
            {
                {{10, 32, 4, 5}, {5, 32, 4, 5}, {1, 32, 4, 5}},
                {{4, 7, 1, 1}, {7, 7, 1, 1}, {1, 7, 1, 1}},
                {{3, 20, 7, 9}, {3, 20, 7, 9}, {6, 20, 7, 9}},
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_byBatch, ConcatLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Values(0),
                                 ::testing::ValuesIn(inputShapes_byBatch),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                                 ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes3D_axis1 = {
        { {}, {{{2, 4, 5}, {2, 4, 5}}} },
        {
            // dynamic
            {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
            // target
            {
                {{2, 5, 12}, {2, 1, 12}, {2, 10, 12}},
                {{1, 16, 1}, {1, 3, 1}, {1, 5, 1}},
                {{5, 2, 6}, {5, 11, 6}, {5, 1, 6}},
            }
        },
        {
            // dynamic
            {{{1, 5}, {2, 16}, {1, 12}}, {{1, 5}, {1, 11}, {1, 21}}, {{1, 5}, {1, 10}, {1, 12}}},
            // target
            {
                {{2, 5, 12}, {2, 1, 12}, {2, 10, 12}},
                {{1, 16, 1}, {1, 3, 1}, {1, 5, 1}},
                {{5, 2, 6}, {5, 11, 6}, {5, 1, 6}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_3D_axis1, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::ValuesIn(inputShapes3D_axis1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes3D_axis2 = {
        { {}, {{{2, 4, 5}, {2, 4, 5}}} },
        {
            // dynamic
            {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
            // target
            {
                {{4, 4, 5}, {4, 4, 1}, {4, 4, 10}},
                {{3, 2, 16}, {3, 2, 3}, {3, 2, 5}},
                {{1, 1, 2}, {1, 1, 11}, {1, 1, 1}},
            }
        },
        {
            // dynamic
            {{{1, 4}, {1, 4}, {2, 16}}, {{1, 4}, {1, 4}, {1, 11}}, {{1, 4}, {1, 4}, {1, 10}}},
            // target
            {
                {{4, 4, 5}, {4, 4, 1}, {4, 4, 10}},
                {{3, 2, 16}, {3, 2, 3}, {3, 2, 5}},
                {{1, 1, 2}, {1, 1, 11}, {1, 1, 1}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_3D_axis2, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::ValuesIn(inputShapes3D_axis2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes2D_axis1 = {
        { {}, {{{3, 2}, {3, 10}}} },
        {
            // dynamic
            {{-1, -1}, {-1, -1}, {-1, -1}},
            // target
            {
                {{19, 5}, {19, 1}, {19, 10}},
                {{1, 16}, {1, 3}, {1, 5}},
                {{8, 2}, {8, 11}, {8, 1}},
            }
        },
        {
            // dynamic
            {{{1, 19}, {2, 16}}, {{1, 19}, {1, 11}}, {{1, 19}, {1, 10}}},
            // target
            {
                {{19, 5}, {19, 1}, {19, 10}},
                {{1, 16}, {1, 3}, {1, 5}},
                {{8, 2}, {8, 11}, {8, 1}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_2D_axis1, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::ValuesIn(inputShapes2D_axis1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        ConcatLayerCPUTest::getTestCaseName);

const std::vector<ConcatInputShapes> inputShapes1D = {
        { {}, {{{5}, {5}}} },
        { {}, {{{2}, {2}}} },
        { {}, {{{1}, {1}}} },
        { {}, {{{3}, {3}}} },
        {
            // dynamic
            {{-1}, {-1}, {-1}},
            // target
            {
                {{19}, {19}, {19}},
                {{8}, {8}, {8}},
                {{5}, {5}, {5}},
            }
        },
        {
            // dynamic
            {{{1, 20}}, {{1, 20}}, {{1, 20}}},
            // target
            {
                {{19}, {19}, {19}},
                {{8}, {8}, {8}},
                {{5}, {5}, {5}},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_1D, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0),
                                ::testing::ValuesIn(inputShapes1D),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                        ConcatLayerCPUTest::getTestCaseName);

// ============================================== inPlace cases ============================================
INSTANTIATE_TEST_SUITE_P(concat_Concat4D_CPU_Block8inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 1),
                                ::testing::Values(ConcatInputShapes{ {}, {{{1, 8, 3, 5}, {1, 8, 3, 5}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D, planarChannels_4D, blocked8_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 1),
                                ::testing::Values(ConcatInputShapes{ {}, {{{1, 32, 3, 5}, {1, 32, 3, 5}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(concat_Concat5D_CPU_Block8inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 1),
                                ::testing::Values(ConcatInputShapes{ {}, {{{1, 16, 3, 5, 7}, {1, 16, 3, 5, 7}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D, planarChannels_5D, blocked8_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block16inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 1),
                                ::testing::Values(ConcatInputShapes{ {}, {{{1, 32, 3, 5, 7}, {1, 32, 3, 5, 7}}} }),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 1, 2),
                                ::testing::ValuesIn(std::vector<ConcatInputShapes>{ConcatInputShapes{ {}, {{{1, 1, 1, 10}, {1, 1, 1, 10}}} },
                                                                                   ConcatInputShapes{ {}, {{{1, 1, 5}, {1, 1, 5}}} }}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                        ConcatLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions