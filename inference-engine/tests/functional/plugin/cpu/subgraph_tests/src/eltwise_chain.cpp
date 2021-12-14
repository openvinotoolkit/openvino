//// Copyright (C) 2018-2021 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include <tuple>
//#include <string>
//#include <vector>
//#include <memory>
//#include <debug.h>
//#include <shared_test_classes/base/layer_test_utils.hpp>
//#include <ngraph_functions/builders.hpp>
//#include <ie_precision.hpp>
//#include "common_test_utils/common_utils.hpp"
//#include "functional_test_utils/precision_utils.hpp"
//#include "functional_test_utils/skip_tests_config.hpp"
//#include "test_utils/cpu_test_utils.hpp"
//#include "ie_system_conf.h"
//
//using namespace CPUTestUtils;
//using InferenceEngine::Precision;
//using ngraph::helpers::EltwiseTypes;
//using FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc;
//
//namespace CPUSubgraphTestsDefinitions {
//
//typedef std::tuple<
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>, // Input shapes
//        ngraph::helpers::InputLayerType,                                                       // Secondary input type
//        std::vector<InferenceEngine::Precision>,                                               // Input precisions
//        std::vector<EltwiseTypes>,                                                             // Eltwise operations
//        bool,                                                                                  // With quantization
//        std::string                                                                            // Device name
//> EltwiseChainTuple;
//
//class EltwiseChainTest : public testing::WithParamInterface<EltwiseChainTuple>,
//                         virtual public LayerTestsUtils::LayerTestsCommon {
//public:
//    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple> &obj) {
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> inputShapes;
//        ngraph::helpers::InputLayerType secondaryInputType;
//        std::vector<InferenceEngine::Precision> inputPrecisions;
//        std::vector<EltwiseTypes> eltwiseOpTypes;
//        bool withQuantization;
//        std::string targetName;
//        std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, targetName) = obj.param;
//        std::ostringstream results;
//
//        results << "IS=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
//        results << "TS=";
//        for (const auto& shape : inputShapes.second) {
//            results << "(";
//            for (const auto& item : shape) {
//                results << CommonTestUtils::vec2str(item) << "_";
//            }
//            results << ")_";
//        }
//        for (int i = 0; i < inputPrecisions.size(); i++) {
//            results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i].name() << "_";
//        }
//        for (int i = 0; i < eltwiseOpTypes.size(); i++) {
//            results << "Op" << std::to_string(i) << "=" << eltwiseOpTypes[i] << "_";
//        }
//        results << "secondaryInputType=" << secondaryInputType << "_";
//        results << "WithQuant=" << withQuantization << "_";
//        results << "targetDevice=" << targetName;
//
//        return results.str();
//    }
//
//    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
//        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 10, 1);
//    }
//
//protected:
//    void SetUp() override {
//        threshold = 0.1f;
//
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> inputShapes;
//        ngraph::helpers::InputLayerType secondaryInputType;
//        std::vector<InferenceEngine::Precision> inputPrecisions;
//        std::vector<EltwiseTypes> eltwiseOpTypes;
//        bool withQuantization;
//        std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, targetDevice) = this->GetParam();
//
//        targetStaticShapes = inputShapes.second;
//        inputDynamicShapes = inputShapes.first;
//
//        ngraph::ParameterVector ngraphParam;
//        std::vector<std::shared_ptr<ngraph::Node>> ngraphInputs;
//        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
//            for (size_t i = 0; i < targetStaticShapes[0].size(); i++) {
//                ngraphParam.push_back(std::make_shared<ngraph::opset1::Parameter>(convertIE2nGraphPrc(inputPrecisions[i]), targetStaticShapes[0][i]));
//                ngraphInputs.push_back(ngraphParam.back());
//            }
//        } else {
//            ngraphParam = ngraph::builder::makeParams(convertIE2nGraphPrc(inputPrecisions[0]), {targetStaticShapes[0][0]});
//            for (int i = 1; i < inputPrecisions.size(); i++) {
//                std::vector<float> ngraphInput1Data(ngraph::shape_size(targetStaticShapes[0][i]));
//                ngraphInputs.push_back(ngraph::builder::makeConstant(convertIE2nGraphPrc(inputPrecisions[i]), targetStaticShapes[0][i],
//                                                                     ngraphInput1Data, true));
//            }
//        }
//
//        if (withQuantization) {
//            std::vector<std::shared_ptr<ngraph::Node>> eltwiseOps;
//            eltwiseOps.push_back(ngraph::builder::makeEltwise(ngraphParam[0], ngraphInputs[0], eltwiseOpTypes[0]));
//            for (int i = 1; i < eltwiseOpTypes.size() - 1; i++) {
//                eltwiseOps.push_back(ngraph::builder::makeEltwise(eltwiseOps[eltwiseOps.size() - 1], ngraphInputs[i], eltwiseOpTypes[i]));
//            }
//
//            std::vector<size_t> constShape(targetStaticShapes[0][0].size(), 1);
//            constShape[1] = targetStaticShapes[0][0][1];
//            auto fq = ngraph::builder::makeFakeQuantize(eltwiseOps[eltwiseOps.size() - 1],
//                                                        ::ngraph::element::Type(::ngraph::element::Type_t::f32),
//                                                        256, constShape);
//
//            eltwiseOps.push_back(ngraph::builder::makeEltwise(fq, ngraphInputs[eltwiseOpTypes.size() - 1], eltwiseOpTypes[eltwiseOpTypes.size() - 1]));
//
//            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseOps[eltwiseOps.size() - 1])};
//            function = std::make_shared<ngraph::Function>(results, ngraphParam, "eltwise_chain_fq");
//        } else {
//            std::vector<std::shared_ptr<ngraph::Node>> eltwiseOps;
//            eltwiseOps.push_back(ngraph::builder::makeEltwise(ngraphParam[0], ngraphInputs[0], eltwiseOpTypes[0]));
//            for (int i = 1; i < eltwiseOpTypes.size(); i++) {
//                eltwiseOps.push_back(ngraph::builder::makeEltwise(eltwiseOps[eltwiseOps.size() - 1], ngraphInputs[i], eltwiseOpTypes[i]));
//            }
//
//            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseOps[eltwiseOps.size() - 1])};
//            function = std::make_shared<ngraph::Function>(results, ngraphParam, "eltwise_chain");
//        }
//    }
//};
//
//TEST_P(EltwiseChainTest, CompareWithRefs) {
//    SKIP_IF_CURRENT_TEST_IS_DISABLED()
//
//    Run();
//}
//
//namespace {
//
//std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inputShapes = {
//    { {}, {{{1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}}}},
//    { {}, {{{1, 48, 5, 6}, {1, 48, 1, 1}, {1, 48, 5, 6}, {1, 1, 5, 6}}}},
//    { {}, {{{1, 72, 28, 28}, {1, 72, 1, 1}, {1, 72, 1, 1}, {1, 72, 1, 1}}}},
//    { {}, {{{2, 33, 5, 5}, {2, 33, 5, 5}, {2, 33, 1, 5}, {2, 33, 5, 5}}}},
//    { {}, {{{1, 2, 3}, {3}, {3}, {3}}}},
//    { {}, {{{1, 12, 5, 5}, {5, 5}, {12, 5, 5}, {1}}}},
//    { {}, {{{3, 12, 5, 5}, {1, 12, 5, 1}, {3, 1, 1, 1}, {3, 12, 5, 5}}}},
//    { {}, {{{1, 1, 1, 1}, {1, 12, 5, 1}, {3, 12, 1, 5}, {3, 12, 5, 1}}}},
//    { {}, {{{1, 1, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}}}}
//};
//
//std::vector<std::vector<InferenceEngine::Precision>> inputPrecisions = {
//        { Precision::FP32, Precision::FP32, Precision::FP32, Precision::FP32 },
//        { Precision::I32, Precision::I32, Precision::I32, Precision::I32 }
//};
//
//std::vector<std::vector<EltwiseTypes>> eltwiseOps = {
//        { EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::SUBTRACT },
//        { EltwiseTypes::DIVIDE, EltwiseTypes::SQUARED_DIFF, EltwiseTypes::ADD },
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain, EltwiseChainTest,
//                        ::testing::Combine(
//                                ::testing::ValuesIn(inputShapes),
//                                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
//                                ::testing::ValuesIn(inputPrecisions),
//                                ::testing::ValuesIn(eltwiseOps),
//                                ::testing::Values(false),
//                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                        EltwiseChainTest::getTestCaseName);
//
//std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inputShapesFQ = {
//    { {}, {{{1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}}}},
//    { {}, {{{2, 33, 5, 5}, {2, 33, 5, 5}, {2, 33, 1, 5}, {2, 33, 5, 5}}}},
//    { {}, {{{2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}}}},
//    { {}, {{{2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}}}},
//    { {}, {{{2, 5, 7, 5}, {2, 5, 1, 5}, {2, 5, 7, 5}, {2, 5, 7, 5}}}},
//    { {}, {{{2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}}}},
//    { {}, {{{2, 256, 7, 5}, {2, 256, 7, 5}, {2, 256, 1, 5}, {2, 256, 7, 5}}}},
//    { {}, {{{1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}}}},
//    { {}, {{{1, 12, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}}}},
//    { {}, {{{1, 12, 1, 1, 6}, {1, 12, 5, 5, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 5, 1}}}},
//    { {}, {{{1, 12, 1, 1, 1}, {1, 12, 5, 1, 7}, {3, 12, 1, 5, 7}, {3, 12, 5, 1, 7}}}},
//    { {}, {{{1, 7, 1, 1, 12}, {1, 7, 5, 1, 12}, {3, 7, 1, 5, 12}, {3, 7, 5, 1, 12}}}},
//    { {}, {{{1, 7, 1, 1, 12, 3, 7}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 3, 7}, {3, 7, 5, 1, 12, 3, 7}}}},
//    { {}, {{{1, 7, 1, 1, 12, 3, 1}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 1, 7}, {3, 7, 5, 1, 12, 3, 1}}}}
//};
//
//std::vector<std::vector<InferenceEngine::Precision>> inputPrecisionsFQ {
//        { Precision::FP32, Precision::FP32, Precision::FP32, Precision::FP32 }
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChainWithFQ, EltwiseChainTest,
//                    ::testing::Combine(
//                            ::testing::ValuesIn(inputShapesFQ),
//                            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
//                            ::testing::ValuesIn(inputPrecisionsFQ),
//                            ::testing::ValuesIn(eltwiseOps),
//                            ::testing::Values(true),
//                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                        EltwiseChainTest::getTestCaseName);
//
//// =============================================== dynamic ==============================================
//std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inputShapes_dyn = {
//    {
//        // dynamic
//        {
//            {-1, -1, -1},
//            {-1},
//            {-1},
//            {-1}
//        },
//        // target
//        {
//            {{1, 2, 3}, {3}, {3}, {3}},
//            {{5, 2, 7}, {7}, {1}, {1}},
//            {{3, 1, 10}, {1}, {1}, {1}},
//        }
//    },
//    {
//        // dynamic
//        {
//            {-1, -1, -1, -1},
//            {-1, -1},
//            {-1, -1, -1},
//            {-1}
//        },
//        // target
//        {
//            {{1, 12, 5, 5}, {5, 5}, {12, 5, 5}, {1}},
//            {{5, 16, 1, 5}, {1, 5}, {1, 5, 1}, {1}},
//            {{2, 1, 1, 5}, {5, 1}, {16, 5, 5}, {5}},
//        }
//    },
//    {
//        // dynamic
//        {
//            {-1, -1, -1, -1},
//            {-1, -1, -1, -1},
//            {-1, -1, -1, -1},
//            {-1, -1, -1, -1}
//        },
//        // target
//        {
//            {{1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}},
//            {{2, 33, 5, 5}, {2, 33, 5, 5}, {2, 33, 1, 5}, {2, 33, 5, 5}},
//            {{2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}},
//            {{2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}},
//            {{2, 5, 7, 5}, {2, 5, 1, 5}, {2, 5, 7, 5}, {2, 5, 7, 5}},
//            {{2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}},
//            {{2, 256, 7, 5}, {2, 256, 7, 5}, {2, 256, 1, 5}, {2, 256, 7, 5}},
//            {{1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}}
//        }
//    },
//    {
//        // dynamic
//        {
//            {-1, -1, -1, -1, -1},
//            {-1, -1, -1, -1, -1},
//            {-1, -1, -1, -1, -1},
//            {-1, -1, -1, -1, -1}
//        },
//        // target
//        {
//            {{1, 12, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}},
//            {{1, 12, 1, 1, 6}, {1, 12, 5, 5, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 5, 1}},
//            {{1, 12, 1, 1, 1}, {1, 12, 5, 1, 7}, {3, 12, 1, 5, 7}, {3, 12, 5, 1, 7}},
//            {{1, 7, 1, 1, 12}, {1, 7, 5, 1, 12}, {3, 7, 1, 5, 12}, {3, 7, 5, 1, 12}}
//        }
//    },
//    {
//        // dynamic
//        {
//            {-1, -1, -1, -1, -1,
//             -1, -1},
//            {-1, -1, -1, -1, -1,
//             -1, -1},
//            {-1, -1, -1, -1, -1,
//             -1, -1},
//            {-1, -1, -1, -1, -1,
//             -1, -1}
//        },
//        // target
//        {
//            {{1, 7, 1, 1, 12, 3, 7}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 3, 7}, {3, 7, 5, 1, 12, 3, 7}},
//            {{1, 7, 1, 1, 12, 3, 1}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 1, 7}, {3, 7, 5, 1, 12, 3, 1}},
//            {{5, 7, 1, 2, 12, 1, 8}, {1, 7, 5, 1, 12, 3, 8}, {5, 1, 1, 2, 12, 1, 8}, {1, 7, 5, 1, 12, 3, 1}}
//        }
//    }
//};
//
//INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_dyn, EltwiseChainTest,
//                        ::testing::Combine(
//                                ::testing::ValuesIn(inputShapes_dyn),
//                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
//                                ::testing::ValuesIn(inputPrecisions),
//                                ::testing::ValuesIn(eltwiseOps),
//                                ::testing::Values(false),
//                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                        EltwiseChainTest::getTestCaseName);
//
//} // namespace
//} // namespace CPUSubgraphTestsDefinitions
