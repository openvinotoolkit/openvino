//// Copyright (C) 2018-2021 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "test_utils/cpu_test_utils.hpp"
//#include "ngraph_functions/builders.hpp"
//
//// Since the Transpose ngraph operation is converted to the transpose node, we will use it in the transpose test
//
//using namespace InferenceEngine;
//using namespace CPUTestUtils;
//
//namespace CPULayerTestsDefinitions {
//
//using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;
//
//typedef std::tuple<
//        inputShapesPair,                    // Input shapes
//        std::vector<size_t>,                // Input order
//        InferenceEngine::Precision,         // Net precision
//        std::string,                        // Target device name
//        std::map<std::string, std::string>, // Additional network configuration
//        CPUSpecificParams> TransposeLayerCPUTestParamSet;
//
//class TransposeLayerCPUTest : public testing::WithParamInterface<TransposeLayerCPUTestParamSet>,
//                              virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
//public:
//    static std::string getTestCaseName(testing::TestParamInfo<TransposeLayerCPUTestParamSet> obj) {
//        Precision netPrecision;
//        inputShapesPair inputShapes;
//        std::vector<size_t> inputOrder;
//        std::string targetDevice;
//        CPUSpecificParams cpuParams;
//        std::map<std::string, std::string> additionalConfig;
//        std::tie(inputShapes, inputOrder, netPrecision, targetDevice, additionalConfig, cpuParams) = obj.param;
//
//        std::ostringstream result;
//        result << "DynShapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
//        result << "StatShapes=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
//        result << "inputOrder=" << CommonTestUtils::vec2str(inputOrder) << "_";
//        result << "netPRC=" << netPrecision.name() << "_";
//        result << "trgDev=" << targetDevice;
//        result << CPUTestsBase::getTestCaseName(cpuParams);
//        return result.str();
//    }
//protected:
//    void SetUp() override {
//        Precision netPrecision;
//        inputShapesPair inputShapes;
//        std::vector<size_t> inputOrder;
//        CPUSpecificParams cpuParams;
//        std::map<std::string, std::string> additionalConfig;
//        std::tie(inputShapes, inputOrder, netPrecision, targetDevice, additionalConfig, cpuParams) = this->GetParam();
//        configuration.insert(additionalConfig.begin(), additionalConfig.end());
//        inPrc = outPrc = netPrecision; // since the layer does not convert precisions
//
//        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
//
//        selectedType = std::string("unknown_") + inPrc.name();
//
//        targetStaticShapes.reserve(inputShapes.second.size());
//        for (const auto& staticShape : inputShapes.second) {
//            targetStaticShapes.push_back({staticShape});
//        }
//        inputDynamicShapes = { inputShapes.first };
//
//        ov::Shape inputDataShape = targetStaticShapes.front().front();
//
//        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
//        auto params = ngraph::builder::makeParams(ngPrc, {inputDataShape});
//        auto paramOuts = ngraph::helpers::convert2OutputVector(
//                ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
//
//        const auto inOrderShape = inputOrder.empty() ? ov::Shape({0}) : ov::Shape({inputDataShape.size()});
//        const auto inputOrderOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
//                                                                         inOrderShape,
//                                                                         inputOrder);
//        const auto transpose = std::make_shared<ov::op::v1::Transpose>(paramOuts.at(0), inputOrderOp);
//        transpose->get_rt_info() = getCPUInfo();
//        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};
//        function = std::make_shared<ov::Model>(results, params, "Transpose");
//    }
//};
//
//TEST_P(TransposeLayerCPUTest, CompareWithRefs) {
//    SKIP_IF_CURRENT_TEST_IS_DISABLED()
//
//    Run();
//    CheckPluginRelatedResults(executableNetwork, "Transpose");
//}
//
//namespace {
//std::map<std::string, std::string> additional_config;
//
//const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {}, {}, {}};
//const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {}, {}, {}};
//
//const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {}, {}, {}};
//const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {}, {}, {}};
//
//const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {}, {}, {}};
//const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {}, {}, {}};
//
//const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {}, {}, {}};
//const auto cpuParams_ncdhw = CPUSpecificParams {{ncdhw}, {}, {}, {}};
//
//const std::vector<InferenceEngine::Precision> netPrecisions = {
//        Precision::I8,
//        Precision::BF16,
//        Precision::FP32
//};
//
//const std::vector<InferenceEngine::Precision> netPrecisionsPerChannels = {
//        Precision::I8,
//        Precision::FP32
//};
//
//const std::vector<inputShapesPair> staticInputShapes4D = {
//        {
//                {},
//                { // Static shapes
//                        {{2, 16, 21, 10}}
//                }
//        },
//        {
//                {},
//                { // Static shapes
//                        {{3, 16, 11, 12}}
//                }
//        },
//        {
//                {},
//                { // Static shapes
//                        {{4, 32, 16, 14}}
//                }
//        },
//        {
//                {},
//                { // Static shapes
//                        {{16, 32, 5, 16}}
//                }
//        }
//};
//const std::vector<inputShapesPair> dynamicInputShapes4D = {
//        {
//                { // Origin dynamic shapes
//                        {ov::Dimension(1, 20), ov::Dimension(10, 40), ov::Dimension(10, 40), ov::Dimension(10, 40)}
//                },
//                { // Dynamic shapes instances
//                        {{1, 32, 21, 10}},
//                        {{2, 25, 11, 12}},
//                        {{4, 15, 16, 14}},
//                        {{7, 10, 20, 16}}
//                }
//        },
//        {
//                { // Origin dynamic shapes
//                        {-1, -1, -1, -1}
//                },
//                { // Dynamic shapes instances
//                        {{1, 24, 21, 8}},
//                        {{2, 16, 11, 6}}
//                }
//        }
//};
//
//const std::vector<std::vector<size_t>> inputOrder4D = {
//        std::vector<size_t>{0, 1, 2, 3},
//        std::vector<size_t>{0, 2, 3, 1},
//        std::vector<size_t>{0, 2, 1, 3},
//        std::vector<size_t>{1, 0, 2, 3},
//        std::vector<size_t>{},
//};
//
//const std::vector<std::vector<size_t>> inputOrderPerChannels4D = {
//        std::vector<size_t>{0, 1, 2, 3},
//        std::vector<size_t>{0, 2, 1, 3},
//        std::vector<size_t>{1, 0, 2, 3},
//        std::vector<size_t>{},
//};
//
//const std::vector<CPUSpecificParams> CPUParams4D = {
//        cpuParams_nChw16c,
//        cpuParams_nChw8c,
//        cpuParams_nchw,
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4D_Transpose, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(staticInputShapes4D),
//                                 ::testing::ValuesIn(inputOrder4D),
//                                 ::testing::ValuesIn(netPrecisions),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::ValuesIn(CPUParams4D)),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_Transpose, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(dynamicInputShapes4D),
//                                 ::testing::ValuesIn(inputOrder4D),
//                                 ::testing::ValuesIn(netPrecisions),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::Values(CPUSpecificParams{})),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4D_PermutePerChannels, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(staticInputShapes4D),
//                                 ::testing::ValuesIn(inputOrderPerChannels4D),
//                                 ::testing::ValuesIn(netPrecisionsPerChannels),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::Values(cpuParams_nhwc)),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_PermutePerChannels, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(dynamicInputShapes4D),
//                                 ::testing::ValuesIn(inputOrderPerChannels4D),
//                                 ::testing::ValuesIn(netPrecisionsPerChannels),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::Values(CPUSpecificParams{})),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//const std::vector<inputShapesPair> staticInputShapes5D = {
//        {
//                {},
//                { // Static shapes
//                        {{2, 16, 5, 6, 5}},
//                        {{3, 16, 6, 5, 6}},
//                        {{4, 32, 5, 6, 5}},
//                        {{5, 32, 6, 5, 6}}
//                }
//        }
//};
//const std::vector<inputShapesPair> dynamicInputShapes5D = {
//        {
//                { // Origin dynamic shapes
//                        {ov::Dimension(1, 20), ov::Dimension(5, 150), ov::Dimension(5, 40), ov::Dimension(5, 40), ov::Dimension(5, 40)}
//                },
//                { // Dynamic shapes instances
//                        {{1, 32, 5, 6, 5}},
//                        {{2, 32, 6, 5, 6}},
//                        {{4, 55, 5, 6, 5}},
//                        {{3, 129, 6, 5, 6}}
//                }
//        }
//};
//
//const std::vector<std::vector<size_t>> inputOrder5D = {
//        std::vector<size_t>{0, 1, 2, 3, 4},
//        std::vector<size_t>{0, 4, 2, 3, 1},
//        std::vector<size_t>{0, 4, 2, 1, 3},
//        std::vector<size_t>{0, 2, 3, 4, 1},
//        std::vector<size_t>{0, 2, 4, 3, 1},
//        std::vector<size_t>{0, 3, 2, 4, 1},
//        std::vector<size_t>{0, 3, 1, 4, 2},
//        std::vector<size_t>{1, 0, 2, 3, 4},
//        std::vector<size_t>{},
//};
//
//const std::vector<std::vector<size_t>> inputOrderPerChannels5D = {
//        std::vector<size_t>{0, 1, 2, 3, 4},
//        std::vector<size_t>{0, 4, 2, 3, 1},
//        std::vector<size_t>{0, 4, 2, 1, 3},
//        std::vector<size_t>{0, 2, 4, 3, 1},
//        std::vector<size_t>{0, 3, 2, 4, 1},
//        std::vector<size_t>{0, 3, 1, 4, 2},
//        std::vector<size_t>{1, 0, 2, 3, 4},
//        std::vector<size_t>{},
//};
//
//const std::vector<CPUSpecificParams> CPUParams5D = {
//        cpuParams_nCdhw16c,
//        cpuParams_nCdhw8c,
//        cpuParams_ncdhw,
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_staticShapes5D_Transpose, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(staticInputShapes5D),
//                                 ::testing::ValuesIn(inputOrder5D),
//                                 ::testing::ValuesIn(netPrecisions),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::ValuesIn(CPUParams5D)),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes5D_Transpose, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(dynamicInputShapes5D),
//                                 ::testing::ValuesIn(inputOrder5D),
//                                 ::testing::ValuesIn(netPrecisions),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::Values(CPUSpecificParams{})),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_staticShapes5D_PermutePerChannels, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(staticInputShapes5D),
//                                 ::testing::ValuesIn(inputOrderPerChannels5D),
//                                 ::testing::ValuesIn(netPrecisionsPerChannels),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::Values(cpuParams_ndhwc)),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes5D_PermutePerChannels, TransposeLayerCPUTest,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(dynamicInputShapes5D),
//                                 ::testing::ValuesIn(inputOrderPerChannels5D),
//                                 ::testing::ValuesIn(netPrecisionsPerChannels),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
//                                 ::testing::Values(additional_config),
//                                 ::testing::Values(CPUSpecificParams{})),
//                         TransposeLayerCPUTest::getTestCaseName);
//
//} // namespace
//} // namespace CPULayerTestsDefinitions
