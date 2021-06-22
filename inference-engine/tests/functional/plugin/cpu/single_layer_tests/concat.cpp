// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        size_t,                            // Concat axis
        std::vector<std::vector<size_t>>,  // Input shapes
        InferenceEngine::Precision,        // Network precision
        std::string,                       // Device name
        CPUSpecificParams
> concatCPUTestParams;

class ConcatLayerCPUTest : public testing::WithParamInterface<concatCPUTestParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatCPUTestParams> obj) {
        int axis;
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        CPUSpecificParams cpuParams;
        std::tie(axis, inputShapes, netPrecision, targetName, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "axis=" << axis << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetName << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        int axis;
        std::vector<std::vector<size_t>> inputShape;
        InferenceEngine::Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(axis, inputShape, netPrecision, targetDevice, cpuParams) = this->GetParam();
        inPrc = outPrc = netPrecision;

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + inPrc.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, inputShape);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);

        function = makeNgraphFunction(ngPrc, params, concat, "concat");
    }
};

TEST_P(ConcatLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Concatenation");
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
const std::vector<Precision> netPrecisions = {
        Precision::I8,
        Precision::I32,
        Precision::FP32,
        Precision::BF16
};

INSTANTIATE_TEST_SUITE_P(concat_Concat4D_CPU_Block8inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::Values(std::vector<std::vector<size_t>>{{1, 8,  3, 5},
                                                                                   {1, 16, 3, 5}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_4D, planarChannels_4D, blocked8_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block8, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 2, 3),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 16, 3, 5},
                                                                                   {2, 16, 3, 5}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_4D_ref, planarChannels_4D, blocked8_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 16, 3, 5},
                                                                                   {2, 32, 3, 5}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_4D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 2, 3),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 32, 3, 5},
                                                                                   {2, 32, 3, 5}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_4D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(concat_Concat5D_CPU_Block8inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::Values(std::vector<std::vector<size_t>>{{1, 8,  3, 5, 7},
                                                                                   {1, 16, 3, 5, 7}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_5D, planarChannels_5D, blocked8_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block8, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 2, 3, 4),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 16, 3, 5, 7},
                                                                                   {2, 16, 3, 5, 7}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_5D_ref, planarChannels_5D, blocked8_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block16inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 16, 3, 5, 7},
                                                                                   {2, 32, 3, 5, 7}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_5D)),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block16, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 2, 3, 4),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 32, 3, 5, 7},
                                                                                   {2, 32, 3, 5, 7}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_5D_ref)),
                        ConcatLayerCPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Concat_inPlace, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(1),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 3, 5},
                                                                                   {2, 4, 5}},
                                                  std::vector<std::vector<size_t>>{{2, 3},
                                                                                   {2, 4}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat3D, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0, 2),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 4, 5},
                                                                                   {2, 4, 5}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_1D_2D, ConcatLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(0),
                                ::testing::Values(std::vector<std::vector<size_t>>{{2, 4},
                                                                                   {3, 4}},
                                                  std::vector<std::vector<size_t>>{{2}, {3}}),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                        ConcatLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions