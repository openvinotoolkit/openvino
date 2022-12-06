// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        InputShape,
        int64_t,              // Axis
        std::vector<int>,     // Split lengths
        ElementType,          // Net precision
        CPUSpecificParams
> varSplitCPUTestParams;

class VariadicSplitLayerCPUTest : public testing::WithParamInterface<varSplitCPUTestParams>,
                                  virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<varSplitCPUTestParams> obj) {
        InputShape shapes;
        int64_t axis;
        std::vector<int> splitLenght;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(shapes, axis, splitLenght, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=";
        result << CommonTestUtils::partialShape2str({shapes.first}) << "_";
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

        InputShape inputShapes;
        int64_t axis;
        std::vector<int> splitLenght;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, axis, splitLenght, netPrecision, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + InferenceEngine::details::convertPrecision(netPrecision).name();

        init_input_shapes({inputShapes});

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
    CheckPluginRelatedResults(compiledModel, "Split");
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

// List of precisions natively supported by onednn.
const std::vector<ElementType> netPrecisions = {
        ElementType::i8,
        ElementType::i32,
        ElementType::f32,
        ElementType::bf16
};

const std::vector<InputShape> inputShapes4D_Nspc2NcspSpecial = {
        { {}, {{3, 5, 24, 9}} },
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 8, 5, 7},
                {3, 9, 7, 9},
                {5, 6, 1, 8}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 64}, {1, 25}, {2, 10}},
            // target
            {
                {2, 7, 5, 7},
                {1, 10, 10, 2},
                {3, 5, 6, 9}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Nspc2NcspSpecial, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_Nspc2NcspSpecial),
                                ::testing::Values(1),
                                ::testing::Values(std::vector<int>{1, 2, -1, 1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(perChannelsToPlanar_4D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_Nspc2NcspSpecial = {
        { {}, {{3, 4, 7, 9, 3}} },
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {1, 6, 5, 7, 5},
                {3, 8, 6, 9, 1},
                {5, 9, 1, 8, 2}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 64}, {1, 25}, {2, 10}, {1, 64}},
            // target
            {
                {2, 5, 5, 7, 7},
                {1, 4, 10, 2, 11},
                {3, 7, 5, 9, 8}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Nspc2NcspSpecial, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_Nspc2NcspSpecial),
                                ::testing::Values(1),
                                ::testing::Values(std::vector<int>{2, 1, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(perChannelsToPlanar_5D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_planar_static, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(InputShape{ {}, {{3, 6, 5, 6}} }),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{1, 3, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D, planar_4D_ref, perChannels_4D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_planar = {
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 9, 8, 7},
                {3, 8, 6, 5},
                {5, 3, 7, 6}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 64}, {1, 48}, {2, 48}},
            // target
            {
                {2, 9, 5, 6},
                {1, 6, 9, 8},
                {3, 1, 6, 7}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_planar, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_planar),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{1, 3, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D_ref, perChannels_4D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_block = {
        { {}, {{3, 16, 6, 7}} },
        {
            // dynamic
            {-1, 16, -1, -1},
            // target
            {
                {1, 16, 8, 7},
                {3, 16, 7, 8},
                {5, 16, 9, 8}
            }
        },
        {
            // dynamic
            {{1, 5}, 16, {1, 48}, {2, 24}},
            // target
            {
                {2, 16, 12, 6},
                {1, 16, 6, 9},
                {3, 16, 7, 6}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Block8, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_block),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{2, 2, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_4D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Block16, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_block),
                                ::testing::Values(2, 3),
                                ::testing::Values(std::vector<int>{2, 2, -1, 1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_4D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_planar_static, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(InputShape{ {}, {{3, 24, 4, 5, 6}} }),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{2, 1, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D, planar_5D_ref, perChannels_5D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_planar = {
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {1, 2, 4, 6, 5},
                {3, 1, 6, 4, 5},
                {5, 6, 5, 7, 4}
            }
        },
        {
            // dynamic
            {{1, 5}, {1, 64}, {1, 48}, {2, 48}, {2, 40}},
            // target
            {
                {2, 5, 4, 5, 6},
                {1, 7, 5, 4, 7},
                {3, 3, 5, 6, 4}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_planar, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_planar),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{2, 1, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_5D_ref, perChannels_5D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_block = {
        { {}, {{3, 16, 8, 5, 6}} },
        {
            // dynamic
            {-1, 16, -1, -1, -1},
            // target
            {
                {1, 16, 5, 6, 7},
                {3, 16, 24, 5, 8},
                {5, 16, 6, 7, 5}
            }
        },
        {
            // dynamic
            {{1, 5}, 16, {1, 48}, {2, 24}, {2, 64}},
            // target
            {
                {2, 16, 7, 6, 5},
                {1, 16, 6, 5, 7},
                {3, 16, 5, 7, 6}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Block8, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_block),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{1, 2, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked8_5D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Block16, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes5D_block),
                                ::testing::Values(2, 3, 4),
                                ::testing::Values(std::vector<int>{2, 1, -1, 1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(blocked16_5D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit3D_static, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(InputShape{ {}, {{14, 7, 21}} }),
                                ::testing::Values(0, 1, 2),
                                ::testing::Values(std::vector<int>{2, 4, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes3D = {
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {7, 21, 14},
                {21, 7, 14},
                {21, 14, 7},
            }
        },
        {
            // dynamic
            {{1, 60}, {1, 50}, {1, 48}},
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
                                ::testing::Values(CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit2D_static, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(InputShape{ {}, {{6, 12}} }),
                                ::testing::Values(0, 1),
                                ::testing::Values(std::vector<int>{2, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes2D = {
        {
            // dynamic
            {-1, -1},
            // target
            {
                {3, 8},
                {10, 4},
                {3, 6},
            }
        },
        {
            // dynamic
            {{1, 60}, {1, 50}},
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
                                ::testing::Values(CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit1D_static, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(InputShape{ {}, {{10}} }),
                                ::testing::Values(0),
                                ::testing::Values(std::vector<int>{2, 1, 1, -1}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes1D = {
        {
            // dynamic
            {-1},
            // target
            {
                {5},
                {15},
                {10},
            }
        },
        {
            // dynamic
            {{1, 60}},
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
                                ::testing::Values(CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_zero_dims = {
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 7, 7, 7},
                {3, 7, 7, 7},
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_zero_dims, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_zero_dims),
                                ::testing::Values(1, 2, 3),
                                ::testing::Values(std::vector<int>{3, 4, -1}, std::vector<int>{3, -1, 4}, std::vector<int>{-1, 3, 4}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(planar_4D_ref)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_zero_dims_nspc_ncsp, VariadicSplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes4D_zero_dims),
                                ::testing::Values(1),
                                ::testing::Values(std::vector<int>{3, 4, -1}, std::vector<int>{3, -1, 4}, std::vector<int>{-1, 3, 4}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(perChannelsToPlanar_4D)),
                        VariadicSplitLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions