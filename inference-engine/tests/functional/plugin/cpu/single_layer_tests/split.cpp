// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        size_t,                         // Num splits
        int64_t,                        // Axis
        InferenceEngine::Precision,     // Net precision
        std::vector<size_t>,            // Input shapes
        std::vector<size_t>,            // Used outputs indices
        std::string,                    // Target device name
        CPUSpecificParams
> splitCPUTestParams;

class SplitLayerCPUTest : public testing::WithParamInterface<splitCPUTestParams>,
                          virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitCPUTestParams> obj) {
        size_t numSplits;
        int64_t axis;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape, outIndices;
        std::string targetDevice;
        CPUSpecificParams cpuParams;
        std::tie(numSplits, axis, netPrecision, inputShape, outIndices, targetDevice, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "numSplits=" << numSplits << "_";
        result << "axis=" << axis << "_";
        if (!outIndices.empty()) {
            result << "outIndices" << CommonTestUtils::vec2str(outIndices) << "_";
        }
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        size_t axis, numSplits;
        std::vector<size_t> inputShape, outIndices;
        InferenceEngine::Precision netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(numSplits, axis, netPrecision, inputShape, outIndices, targetDevice, cpuParams) = this->GetParam();
        inPrc = outPrc = netPrecision;
        if (outIndices.empty()) {
            for (int i = 0; i < numSplits; ++i) {
                outIndices.push_back(i);
            }
        }

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + inPrc.name();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto split = std::dynamic_pointer_cast<ngraph::opset5::Split>(ngraph::builder::makeSplit(paramOuts[0],
                                                                                                 ngPrc, numSplits, axis));
        ngraph::ResultVector results;

        for (int i = 0; i < outIndices.size(); i++) {
            // This WA is necessary because result nodes connected to the same output of the split node (or any node) are deduplicated
            // on the CNNNetwork level. It might not be needed when the CPU plugin moves completely to nGraph.
            // This is still a single layer test since the Ceiling nodes are added only as a WA.

            auto fakeMultiplication = std::make_shared<ngraph::opset5::Ceiling>(split->output(outIndices[i]));
            results.push_back(std::make_shared<ngraph::opset5::Result>(fakeMultiplication));
        }
        split->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(results, params, "split");
    }
};

TEST_P(SplitLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Split");
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
const std::vector<Precision> netPrecisions = {
        Precision::I8,
        Precision::I32,
        Precision::FP32,
        Precision::BF16
};

const std::vector<std::vector<size_t>> outIndices3 = {{0, 1, 2}, {0, 1, 1, 0, 2}, {0, 0, 0, 2}};
const std::vector<std::vector<size_t>> outIndices4 = {{0, 1, 2, 3}, {0, 1, 1, 0, 2, 3}, {0, 0, 0, 2, 3}};


INSTANTIATE_TEST_CASE_P(smoke_Split4D_CPU_Nspc2NcspSpecial, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::Values(1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({3, 28, 24, 9})),
                                ::testing::ValuesIn(outIndices4),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(perChannelsToPlanar_4D)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split5D_CPU_Nspc2NcspSpecial, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::Values(1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({3, 21, 24, 9, 15})),
                                ::testing::ValuesIn(outIndices3),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(perChannelsToPlanar_5D)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split4D_CPU_Block8inPlace, SplitLayerCPUTest,
                    ::testing::Combine(
                            ::testing::Values(3),
                            ::testing::Values(0, 1),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(std::vector<size_t>({3, 24, 24, 9})),
                            ::testing::ValuesIn(outIndices3),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                            ::testing::Values(planar_4D, planar_4D_ref, perChannels_4D, blocked8_4D)),
                    SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split4D_CPU_Block8, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::Values(2, 3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({3, 24, 24, 9})),
                                ::testing::ValuesIn(outIndices3),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_4D, planar_4D_ref, perChannels_4D, blocked8_4D_ref)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split4D_CPU_Block16inPlace, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::Values(0, 1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({4, 64, 32, 12})),
                                ::testing::ValuesIn(outIndices3),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_4D)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split4D_CPU_Block16, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::Values(2, 3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({4, 64, 32, 12})),
                                ::testing::ValuesIn(outIndices4),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_4D_ref)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split5D_CPU_Block8inPlace, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::Values(0, 1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({3, 24, 24, 9, 15})),
                                ::testing::ValuesIn(outIndices3),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_5D, planar_5D_ref, perChannels_5D, blocked8_5D)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split5D_CPU_Block8, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(3),
                                ::testing::Values(2, 3, 4),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({3, 24, 24, 9, 15})),
                                ::testing::ValuesIn(outIndices3),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(planar_5D, planar_5D_ref, perChannels_5D, blocked8_5D_ref)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split5D_CPU_Block16inPlace, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::Values(0, 1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({4, 64, 32, 12, 20})),
                                ::testing::ValuesIn(outIndices4),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_5D)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split5D_CPU_Block16, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(4),
                                ::testing::Values(2, 3, 4),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({4, 64, 32, 12, 20})),
                                ::testing::ValuesIn(outIndices4),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(blocked16_5D_ref)),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split3D, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(7),
                                ::testing::Values(0, 1, 2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({14, 42, 21})),
                                ::testing::Values(std::vector<size_t>({})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                                SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split2D, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(2),
                                ::testing::Values(0, 1),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({6, 12})),
                                ::testing::Values(std::vector<size_t>({})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                        SplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Split1D, SplitLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Values(5),
                                ::testing::Values(0),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t>({10})),
                                ::testing::Values(std::vector<size_t>({})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"}, CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                            SplitLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions