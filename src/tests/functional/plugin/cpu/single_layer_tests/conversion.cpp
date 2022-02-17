//// Copyright (C) 2018-2022 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "shared_test_classes/base/layer_test_utils.hpp"
//#include "test_utils/cpu_test_utils.hpp"
//#include "ngraph_functions/builders.hpp"
//
//using namespace InferenceEngine;
//using namespace ngraph;
//using namespace CPUTestUtils;
//
//namespace CPULayerTestsDefinitions {
//
//using convertLayerShapeDefinition = std::pair<std::vector<ngraph::PartialShape>, std::vector<ngraph::Shape>>;
//
//using convertLayerTestParamsSet = std::tuple<convertLayerShapeDefinition,  // input shapes
//                                        InferenceEngine::Precision,        // input precision
//                                        InferenceEngine::Precision,        // output precision
//                                        CPUSpecificParams>;
//
//class ConvertCPULayerTest : public testing::WithParamInterface<convertLayerTestParamsSet>,
//                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
//public:
//    static std::string getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj) {
//        convertLayerShapeDefinition shapes;
//        InferenceEngine::Precision inPrc, outPrc;
//        CPUSpecificParams cpuParams;
//        std::tie(shapes, inPrc, outPrc, cpuParams) = obj.param;
//
//        std::ostringstream result;
//        if (!shapes.first.empty()) {
//            result << "IS=" << CommonTestUtils::partialShape2str(shapes.first) << "_";
//        }
//        result << "TS=";
//        for (const auto& shape : shapes.second) {
//            result << CommonTestUtils::vec2str(shape) << "_";
//        }
//        result << "inputPRC=" << inPrc.name() << "_";
//        result << "targetPRC=" << outPrc.name() << "_";
//        result << CPUTestsBase::getTestCaseName(cpuParams);
//
//        return result.str();
//    }
//
//protected:
//    void SetUp() override {
//        targetDevice = CommonTestUtils::DEVICE_CPU;
//
//        convertLayerShapeDefinition shapes;
//        InferenceEngine::Precision inPrc, outPrc;
//        CPUSpecificParams cpuParams;
//        std::tie(shapes, inPrc, outPrc, cpuParams) = GetParam();
//
//        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
//
//        selectedType = std::string("unknown_") + (inPrc == InferenceEngine::Precision::U8 ? "I8" : inPrc.name());
//
//        for (size_t i = 0; i < shapes.second.size(); i++) {
//            targetStaticShapes.push_back(std::vector<ngraph::Shape>{shapes.second[i]});
//        }
//        inputDynamicShapes = shapes.first;
//
//        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
//        auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
//        auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShapes[0][0]});
//        auto conversion = ngraph::builder::makeConversion(params.front(), targetPrc, helpers::ConversionTypes::CONVERT);
//
//        function = makeNgraphFunction(ngPrc, params, conversion, "ConversionCPU");
//    }
//};
//
//TEST_P(ConvertCPULayerTest, CompareWithRefs) {
//    SKIP_IF_CURRENT_TEST_IS_DISABLED()
//
//    Run();
//
//    CheckPluginRelatedResults(executableNetwork, "Convert");
//}
//
//std::vector<convertLayerShapeDefinition> inShapes_4D = {
//        {{}, {{1, 2, 3, 4}}},
//        {
//            // dynamic
//            {{-1, -1, -1, -1}},
//            // target
//            {
//                {2, 4, 4, 1},
//                {2, 17, 5, 4},
//                {1, 2, 3, 4}
//            }
//        },
//        {
//            // dynamic
//            {{{1, 5}, {2, 22}, {2, 9}, {1, 4}}},
//            // target
//            {
//                {2, 17, 5, 4},
//                {5, 2, 3, 2},
//                {1, 10, 4, 1},
//            }
//        }
//};
//
//// List of precisions natively supported by mkldnn.
//const std::vector<Precision> precisions = {
//        Precision::U8,
//        Precision::I8,
//        Precision::I32,
//        Precision::FP32,
//        Precision::BF16
//};
//
//std::vector<CPUSpecificParams> memForm4D = {
//        CPUSpecificParams({nchw}, {nchw}, {}, {}),
//        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
//        CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}),
//        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest, ConvertCPULayerTest,
//                        ::testing::Combine(
//                                ::testing::ValuesIn(inShapes_4D),
//                                ::testing::ValuesIn(precisions),
//                                ::testing::ValuesIn(precisions),
//                                ::testing::ValuesIn(memForm4D)),
//                        ConvertCPULayerTest::getTestCaseName);
//
//} // namespace CPULayerTestsDefinitions