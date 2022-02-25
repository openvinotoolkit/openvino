// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
// Since the Transpose ngraph operation is converted to the transpose node, we will use it in the transpose test

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        InputShape,                    // Input shapes
        std::vector<size_t>,                // Input order
        InferenceEngine::Precision,         // Net precision
        std::string,                        // Target device name
        std::map<std::string, std::string>, // Additional network configuration
        CPUSpecificParams> TransposeLayerCPUTestParamSet;

class TransposeLayerCPUTest : public testing::WithParamInterface<TransposeLayerCPUTestParamSet>,
                              public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeLayerCPUTestParamSet> obj) {
        Precision netPrecision;
        InputShape inputShapes;
        std::vector<size_t> inputOrder;
        std::string targetDevice;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, inputOrder, netPrecision, targetDevice, additionalConfig, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::partialShape2str({inputShapes.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShapes.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "inputOrder=" << CommonTestUtils::vec2str(inputOrder) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        Precision netPrecision;
        InputShape inputShapes;
        std::vector<size_t> inputOrder;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, inputOrder, netPrecision, targetDevice, additionalConfig, cpuParams) = this->GetParam();
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        outType[0] = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType = makeSelectedTypeStr("unknown", inType);

        init_input_shapes({inputShapes});

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeDynamicParams(inType, { inputDynamicShapes[0] });

        const auto inputOrderOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                         ov::Shape({inputOrder.size()}),
                                                                         inputOrder);
        const auto transpose = std::make_shared<ov::op::v1::Transpose>(params[0], inputOrderOp);
        transpose->get_rt_info() = getCPUInfo();
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};

        function = std::make_shared<ngraph::Function>(results, params, "TransposeLayerCPUTest");
        functionRefs = ngraph::clone_function(*function);
    }
};

TEST_P(TransposeLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Transpose");
}

namespace {
std::map<std::string, std::string> additional_config;

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams {{ncdhw}, {}, {}, {}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        Precision::I8,
        Precision::BF16,
        Precision::FP32
};

const std::vector<InferenceEngine::Precision> netPrecisionsPerChannels = {Precision::I8, Precision::FP32};

const std::vector<InputShape> staticInputShapes4DC16 = {InputShape{// dynamic
                                                                   {-1, 16, -1, -1},
                                                                   // target
                                                                   {{2, 16, 21, 10}, {3, 16, 11, 12}, {2, 16, 21, 10}}}};

const std::vector<InputShape> staticInputShapes4DC32 = {InputShape{// dynamic
                                                                   {-1, 32, -1, -1},
                                                                   // target
                                                                   {{4, 32, 16, 14}, {16, 32, 5, 16}, {4, 32, 16, 14}}}};

const std::vector<InputShape> dynamicInputShapes4D = {
    InputShape{// dynamic
               {ov::Dimension(1, 20), ov::Dimension(10, 40), ov::Dimension(10, 40), ov::Dimension(10, 40)},
               // target
               {{1, 32, 21, 10}, {2, 25, 11, 12}, {4, 15, 16, 14}, {7, 10, 20, 16}, {1, 32, 21, 10}}},
    InputShape{// dynamic
               {-1, -1, -1, -1},
               // target
               {{1, 24, 21, 8}, {2, 16, 11, 6}, {1, 24, 21, 8}}}
};

const std::vector<std::vector<size_t>> inputOrder4D = {
        std::vector<size_t>{0, 1, 2, 3},
        std::vector<size_t>{0, 2, 3, 1},
        std::vector<size_t>{0, 2, 1, 3},
        std::vector<size_t>{1, 0, 2, 3},
        std::vector<size_t>{},
};

const std::vector<std::vector<size_t>> inputOrderPerChannels4D = {
        std::vector<size_t>{0, 1, 2, 3},
        std::vector<size_t>{0, 2, 1, 3},
        std::vector<size_t>{1, 0, 2, 3},
        std::vector<size_t>{},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nchw,
};

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes4DC16),
                                 ::testing::ValuesIn(inputOrder4D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes4DC32),
                                 ::testing::ValuesIn(inputOrder4D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D),
                                 ::testing::ValuesIn(inputOrder4D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes4DC16),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes4DC32),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

const std::vector<InputShape> staticInputShapes5DC16 = {InputShape{
    // dynamic
    {-1, 16, -1, -1, -1},
    // Static shapes
    {{2, 16, 5, 6, 5}, {3, 16, 6, 5, 6}, {2, 16, 5, 6, 5}}}
};

const std::vector<InputShape> staticInputShapes5DC32 = {InputShape{
    // dynamic
    {-1, 32, -1, -1, -1},
    // Static shapes
    {{4, 32, 5, 6, 5}, {5, 32, 6, 5, 6}, {4, 32, 5, 6, 5}}}
};

const std::vector<InputShape> dynamicInputShapes5D = {InputShape{
    // dynamic
    {ov::Dimension(1, 20), ov::Dimension(5, 150), ov::Dimension(5, 40), ov::Dimension(5, 40), ov::Dimension(5, 40)},
    // target
    {{1, 32, 5, 6, 5}, {2, 32, 6, 5, 6}, {4, 55, 5, 6, 5}, {3, 129, 6, 5, 6}, {1, 32, 5, 6, 5}}}
};

const std::vector<std::vector<size_t>> inputOrder5D = {
        std::vector<size_t>{0, 1, 2, 3, 4},
        std::vector<size_t>{0, 4, 2, 3, 1},
        std::vector<size_t>{0, 4, 2, 1, 3},
        std::vector<size_t>{0, 2, 3, 4, 1},
        std::vector<size_t>{0, 2, 4, 3, 1},
        std::vector<size_t>{0, 3, 2, 4, 1},
        std::vector<size_t>{0, 3, 1, 4, 2},
        std::vector<size_t>{1, 0, 2, 3, 4},
        std::vector<size_t>{},
};

const std::vector<std::vector<size_t>> inputOrderPerChannels5D = {
        std::vector<size_t>{0, 1, 2, 3, 4},
        std::vector<size_t>{0, 4, 2, 3, 1},
        std::vector<size_t>{0, 4, 2, 1, 3},
        std::vector<size_t>{0, 2, 4, 3, 1},
        std::vector<size_t>{0, 3, 2, 4, 1},
        std::vector<size_t>{0, 3, 1, 4, 2},
        std::vector<size_t>{1, 0, 2, 3, 4},
        std::vector<size_t>{},
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ncdhw,
};

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes5DC16_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes5DC16),
                                 ::testing::ValuesIn(inputOrder5D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::ValuesIn(CPUParams5D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes5DC32_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes5DC32),
                                 ::testing::ValuesIn(inputOrder5D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::ValuesIn(CPUParams5D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes5D_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes5D),
                                 ::testing::ValuesIn(inputOrder5D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes5DC16_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes5DC16),
                                 ::testing::ValuesIn(inputOrderPerChannels5D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(cpuParams_ndhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes5DC32_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShapes5DC32),
                                 ::testing::ValuesIn(inputOrderPerChannels5D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(cpuParams_ndhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes5D_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes5D),
                                 ::testing::ValuesIn(inputOrderPerChannels5D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
