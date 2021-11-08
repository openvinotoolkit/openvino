// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using VarSplitInputShapes = std::pair<std::vector<ov::PartialShape>, std::vector<ov::Shape>>;

typedef std::tuple<
        VarSplitInputShapes,
        int64_t,              // Axis
        std::vector<int>,     // Split lengths
        ElementType,          // Net precision
        CPUSpecificParams
> varSplitCPUTestParams;

class VariadicSplitLayerCPUTest : public testing::WithParamInterface<varSplitCPUTestParams>,
                                  virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<varSplitCPUTestParams> obj) {
        VarSplitInputShapes shapes;
        int64_t axis;
        std::vector<int> splitLenght;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(shapes, axis, splitLenght, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        if (!shapes.first.empty()) {
            auto graphShape = shapes.first.front();
            result << CommonTestUtils::partialShape2str(shapes.first) << ")_";
        }
        result << "TS=";
        for (const auto& shape : shapes.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << "axis=" << axis << "_";
        result << "splitLenght=" << CommonTestUtils::vec2str(splitLenght) << "_";
        result << "netPRC=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        VarSplitInputShapes inputShapes;
        int64_t axis;
        std::vector<int> splitLenght;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, axis, splitLenght, netPrecision, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + InferenceEngine::details::convertPrecision(netPrecision).name();

        if (!inputShapes.first.empty()) {
            inputDynamicShapes = inputShapes.first;
        } else {
            inputDynamicShapes = {inputShapes.second.front()};
        }

        for (const auto &td : inputShapes.second) {
            targetStaticShapes.push_back({td});
        }

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto splitAxisOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t>{axis});
        auto splitLengthsOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32, ngraph::Shape{splitLenght.size()}, splitLenght);
        auto varSplit = std::make_shared<ngraph::opset3::VariadicSplit>(paramOuts[0], splitAxisOp, splitLengthsOp);

        varSplit->get_rt_info() = getCPUInfo();

        function = std::make_shared<ngraph::Function>(varSplit, params, "VariadicSplitCPU");
    }
};

TEST_P(VariadicSplitLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
//     CheckPluginRelatedResults(executableNetwork, "Split");
}

namespace {
const auto planar_4D_ref = CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref"};
const auto planar_5D_ref = CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref"}, "ref"};

const auto planar_4D = CPUSpecificParams{{nchw}, {nchw}, {}, "unknown"};
const auto planar_5D = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "unknown"};

const auto perChannels_4D = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto perChannels_5D = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};

const auto perChannelsToPlanar_4D = CPUSpecificParams{{nhwc}, {nchw}, {}, "ref"};
const auto perChannelsToPlanar_5D = CPUSpecificParams{{ndhwc}, {ncdhw}, {}, "ref"};

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

const std::vector<VarSplitInputShapes> inputShapes4D_Nspc2NcspSpecial = {
        { {}, {{3, 28, 24, 9}} },
        {
            // dynamic
            {{-1, -1, -1, -1}},
            // target
            {
                {1, 16, 5, 7},
                {3, 28, 24, 9},
                {5, 12, 1, 8}
            }
        },
        {
            // dynamic
            {{{1, 5}, {1, 64}, {1, 25}, {2, 10}}},
            // target
            {
                {2, 64, 5, 7},
                {1, 8, 10, 2},
                {3, 28, 24, 9}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Nspc2NcspSpecial, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_Nspc2NcspSpecial),
                                ::testing::Values(1),
                                ::testing::Values(std::vector<int>{2, 3, -1, 1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(perChannelsToPlanar_4D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes5D_Nspc2NcspSpecial = {
        { {}, {{3, 21, 24, 9, 15}} },
        {
            // dynamic
            {{-1, -1, -1, -1, -1}},
            // target
            {
                {1, 12, 5, 7, 5},
                {3, 27, 24, 9, 1},
                {5, 12, 1, 8, 2}
            }
        },
        {
            // dynamic
            {{{1, 5}, {1, 64}, {1, 25}, {2, 10}, {1, 64}}},
            // target
            {
                {2, 60, 5, 7, 7},
                {1, 7, 10, 2, 11},
                {3, 27, 24, 9, 8}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Nspc2NcspSpecial, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_Nspc2NcspSpecial),
                                ::testing::Values(1),
                                ::testing::Values(std::vector<int>{3, 3, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(perChannelsToPlanar_5D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes4D_planar = {
        { {}, {{3, 24, 15, 11}} },
        {
            // dynamic
            {{-1, -1, -1, -1}},
            // target
            {
                {1, 48, 12, 15},
                {3, 12, 24, 11},
                {5, 24, 24, 23}
            }
        },
        {
            // dynamic
            {{{1, 5}, {1, 64}, {1, 48}, {2, 48}}},
            // target
            {
                {2, 9, 12, 48},
                {1, 6, 12, 12},
                {3, 1, 48, 11}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_planar, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_planar),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{5, 5, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D, planar_4D_ref, perChannels_4D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes4D_block = {
        { {}, {{3, 32, 24, 12}} },
        {
            // dynamic
            {{-1, 48, -1, -1}},
            // target
            {
                {1, 48, 12, 48},
                {3, 48, 24, 12},
                {5, 48, 24, 12}
            }
        },
        {
            // dynamic
            {{{1, 5}, 48, {1, 48}, {2, 24}}},
            // target
            {
                {2, 48, 12, 12},
                {1, 48, 12, 24},
                {3, 48, 48, 24}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Block8, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_block),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{5, 5, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_4D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Block16, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_block),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{5, 5, -1, 1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_4D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes5D_planar = {
        { {}, {{3, 24, 24, 11, 15}} },
        {
            // dynamic
            {{-1, -1, -1, -1, -1}},
            // target
            {
                {1, 15, 12, 48, 15},
                {3, 1, 24, 12, 30},
                {5, 23, 24, 24, 24}
            }
        },
        {
            // dynamic
            {{{1, 5}, {1, 64}, {1, 48}, {2, 48}, {10, 40}}},
            // target
            {
                {2, 5, 12, 48, 24},
                {1, 7, 12, 11, 15},
                {3, 11, 48, 11, 30}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_planar, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_planar),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{5, 5, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D, planar_5D_ref, perChannels_5D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes5D_block = {
        { {}, {{3, 32, 24, 12, 36}} },
        {
            // dynamic
            {{-1, 48, -1, -1, -1}},
            // target
            {
                {1, 48, 12, 48, 36},
                {3, 48, 24, 12, 12},
                {5, 48, 24, 12, 24}
            }
        },
        {
            // dynamic
            {{{1, 5}, 48, {1, 48}, {2, 24}, {12, 64}}},
            // target
            {
                {2, 48, 12, 12, 24},
                {1, 48, 12, 24, 36},
                {3, 48, 48, 24, 12}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Block8, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_block),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{5, 5, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_5D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Block16, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_block),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{5, 5, -1, 1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_5D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes3D = {
        { {}, {{14, 42, 21}} },
        {
            // dynamic
            {{-1, -1, -1}},
            // target
            {
                {7, 21, 14},
                {21, 7, 14},
                {21, 14, 7},
            }
        },
        {
            // dynamic
            {{{1, 60}, {1, 50}, {1, 48}}},
            // target
            {
                {14, 21, 7},
                {21, 7, 14},
                {7, 14, 21},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit3D, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::Values(0, 1, 2),
                                ::testing::Values(std::vector<int>{2, 4, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes2D = {
        { {}, {{6, 12}} },
        {
            // dynamic
            {{-1, -1}},
            // target
            {
                {3, 8},
                {10, 4},
                {3, 6},
            }
        },
        {
            // dynamic
            {{{1, 60}, {1, 50}}},
            // target
            {
                {3, 4},
                {4, 4},
                {6, 12},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit2D, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(0, 1),
                                ::testing::Values(std::vector<int>{2, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<VarSplitInputShapes> inputShapes1D = {
        { {}, {{10}} },
        {
            // dynamic
            {{-1}},
            // target
            {
                {5},
                {15},
                {10},
            }
        },
        {
            // dynamic
            {{{1, 60}}},
            // target
            {
                {15},
                {5},
                {10},
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit1D, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes1D),
                                ::testing::Values(0),
                                ::testing::Values(std::vector<int>{2, 1, 1, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions