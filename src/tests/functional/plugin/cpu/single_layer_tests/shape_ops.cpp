//// Copyright (C) 2018-2022 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "test_utils/cpu_test_utils.hpp"
//#include "ngraph_functions/builders.hpp"
//
//using namespace ngraph;
//using namespace InferenceEngine;
//using namespace CPUTestUtils;
//
//namespace CPULayerTestsDefinitions {
//
//enum class shapeNodeType {
//    Reshape,
//    Squeeze,
//    Unsqueeze
//};
//
//std::ostream& operator<<(std::ostream & os, shapeNodeType type) {
//    switch (type) {
//        case shapeNodeType::Reshape:
//            os << "Reshape";
//            break;
//        case shapeNodeType::Squeeze:
//            os << "Squeeze";
//            break;
//        case shapeNodeType::Unsqueeze:
//            os << "Unsqueeze";
//            break;
//    }
//    return os;
//}
//
//struct inferShapes {
//    ngraph::PartialShape dynShape;
//    std::vector<ngraph::Shape> inferShape;
//    std::vector<std::vector<int>> data;
//};
//
//using shapeOpsParams = std::tuple<
//    inferShapes,                       // input shapes
//    ngraph::helpers::InputLayerType,   // second input type
//    shapeNodeType,                     // node type
//    Precision,                         // precision
//    bool>;                             // special zero
//
//class ShapeOpsCPUTest : public testing::WithParamInterface<shapeOpsParams>, virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
//public:
//    static std::string getTestCaseName(testing::TestParamInfo<shapeOpsParams> obj) {
//        inferShapes shapes;
//        ngraph::helpers::InputLayerType secondType;
//        shapeNodeType nodeType;
//        Precision prc;
//        bool specialZero;
//        std::tie(shapes, secondType, nodeType, prc, specialZero) = obj.param;
//
//        std::ostringstream result;
//        result << nodeType << "_";
//        result << "IS=" << CommonTestUtils::partialShape2str({shapes.dynShape}) << "_";
//        result << "TS=";
//        for (const auto& shape : shapes.inferShape) {
//            result << "(" << CommonTestUtils::vec2str(shape) << ")_";
//        }
//        result << "DATA=";
//        for (const auto& data : shapes.data) {
//            result << "[" << CommonTestUtils::vec2str(data) << "]_";
//        }
//        result << "PRC=" << prc << "_";
//        result << "specialZero=" << specialZero;
//
//        return result.str();
//    }
//
//protected:
//    void GenerateInputs() override {
//        inputs.clear();
//        const auto& inputsInfo = executableNetwork.GetInputsInfo();
//        const auto& functionParams = function->get_parameters();
//        for (int i = 0; i < functionParams.size(); ++i) {
//            const auto& param = functionParams[i];
//            const auto infoIt = inputsInfo.find(param->get_friendly_name());
//            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
//            InferenceEngine::InputInfo::CPtr info = infoIt->second;
//            InferenceEngine::DataPtr dataNew(
//                        new InferenceEngine::Data(infoIt->first, info->getTensorDesc().getPrecision(),
//                                                  targetStaticShapes[index][i],
//                                                  info->getTensorDesc().getLayout()));
//            InferenceEngine::InputInfo infoNew;
//            infoNew.setInputData(dataNew);
//
//            InferenceEngine::Blob::Ptr blob;
//            if (i == 1) {
//                blob = make_blob_with_precision(infoNew.getTensorDesc(), data[index].data());
//            } else {
//                blob = GenerateInput(infoNew);
//            }
//            inputs.push_back(blob);
//        }
//    }
//
//    void SetUp() override {
//        targetDevice = CommonTestUtils::DEVICE_CPU;
//
//        inferShapes shapes;
//        ngraph::helpers::InputLayerType secondType;
//        shapeNodeType nodeType;
//        Precision prc;
//        bool specialZero;
//        std::tie(shapes, secondType, nodeType, prc, specialZero) = this->GetParam();
//
//        selectedType = std::string("unknown_") + prc.name();
//
//        data = shapes.data;
//
//        for (size_t i = 0; i < shapes.inferShape.size(); i++) {
//            targetStaticShapes.push_back(std::vector<ov::Shape>{shapes.inferShape[i], {shapes.data[0].size()}});
//        }
//        inputDynamicShapes = {shapes.dynShape};
//
//        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);
//        const auto secondInPrc = ngraph::element::Type_t::i32;
//        ngraph::ParameterVector inputs;
//        const auto dataInput = std::make_shared<ngraph::opset1::Parameter>(ngPrc, targetStaticShapes.front().front());
//        dataInput->set_friendly_name("param_1");
//        inputs.push_back(dataInput);
//        std::shared_ptr<ngraph::Node> secondaryInput;
//        if (secondType == ngraph::helpers::InputLayerType::PARAMETER) {
//            secondaryInput = std::make_shared<ngraph::opset1::Parameter>(secondInPrc, ngraph::Shape({shapes.data[0].size()}));
//            secondaryInput->set_friendly_name("param_2");
//            inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
//            inputDynamicShapes.push_back(ngraph::Shape({shapes.data[0].size()}));
//        } else {
//            secondaryInput = ngraph::builder::makeConstant(secondInPrc, {shapes.data[0].size()}, shapes.data[0]);
//        }
//
//        std::shared_ptr<ngraph::Node> shapeOps;
//        switch (nodeType) {
//            case shapeNodeType::Reshape: {
//                shapeOps = std::make_shared<ngraph::opset1::Reshape>(dataInput, secondaryInput, specialZero);
//                break;
//            }
//            case shapeNodeType::Squeeze: {
//                shapeOps = std::make_shared<ngraph::opset1::Squeeze>(dataInput, secondaryInput);
//                break;
//            }
//            case shapeNodeType::Unsqueeze: {
//                shapeOps = std::make_shared<ngraph::opset1::Unsqueeze>(dataInput, secondaryInput);
//                break;
//            }
//        }
//
//        function = std::make_shared<ngraph::Function>(shapeOps, inputs, "ShapeOpsCPUTest");
//        functionRefs = ngraph::clone_function(*function);
//    }
//
//private:
//    std::vector<std::vector<int>> data;
//};
//
//TEST_P(ShapeOpsCPUTest, CompareWithRefs) {
//    SKIP_IF_CURRENT_TEST_IS_DISABLED()
//
//    Run();
//    CheckPluginRelatedResults(executableNetwork, "Reshape");
//}
//
//namespace reshapeTest {
//
//inferShapes noBounds{{-1, -1, -1, -1},
//                     {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{1, 2, 5, 5}},
//                     {std::vector<int>{1, -1, 0}, std::vector<int>{-1, 60, 2}, std::vector<int>{10, 30, 10}, std::vector<int>{5, 10, -1}}};
//
//const auto params = ::testing::Combine(::testing::Values(noBounds),
//                                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
//                                       ::testing::Values(shapeNodeType::Reshape),
//                                       ::testing::Values(Precision::FP32),
//                                       ::testing::Values(true));
//
//INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, ShapeOpsCPUTest, params, ShapeOpsCPUTest::getTestCaseName);
//
//inferShapes noBounds_const{{{1, 10}, {2, 6}, {1, 15}, {3, 11}},
//                           {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{1, 2, 5, 5}},
//                           {std::vector<int>{2, -1, 0}}};
//
//const auto params_const = ::testing::Combine(::testing::Values(noBounds_const),
//                                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
//                                             ::testing::Values(shapeNodeType::Reshape),
//                                             ::testing::Values(Precision::FP32),
//                                             ::testing::Values(true));
//
//INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_const, ShapeOpsCPUTest, params_const, ShapeOpsCPUTest::getTestCaseName);
//
//} // namespace reshapeTest
//
//namespace squeezeTest {
//
//inferShapes noBounds{{-1, -1, -1, -1, -1, -1},
//                     {ngraph::Shape{2, 5, 1, 7, 3, 1}, ngraph::Shape{10, 1, 1, 6, 10, 5}, ngraph::Shape{10, 6, 10, 5, 1, 1}, ngraph::Shape{1, 1, 5, 1, 5}},
//                     {std::vector<int>{2, 5}, std::vector<int>{1, 2}, std::vector<int>{4, 5}, std::vector<int>{0, 1}}};
//
//const auto params = ::testing::Combine(::testing::Values(noBounds),
//                                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
//                                       ::testing::Values(shapeNodeType::Squeeze),
//                                       ::testing::Values(Precision::FP32),
//                                       ::testing::Values(true));
//
//// at this momemnt squeze produce dynamic output rank, if second input is not constant
//// enable after CPU plug-in will support dynamic rank
//// INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, ShapeOpsCPUTest, params, ShapeOpsCPUTest::getTestCaseName);
//
//inferShapes noBounds_const{{{1, 10}, {1, 15}, {2, 6}, {1, 15}, {3, 11}, {1, 15}},
//                           {ngraph::Shape{2, 1, 5, 7, 3, 1}, ngraph::Shape{10, 1, 6, 10, 5, 1}, ngraph::Shape{1, 1, 2, 5, 5, 1}},
//                           {std::vector<int>{1, 5}}};
//
//const auto params_const = ::testing::Combine(::testing::Values(noBounds_const),
//                                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
//                                             ::testing::Values(shapeNodeType::Squeeze),
//                                             ::testing::Values(Precision::FP32),
//                                             ::testing::Values(true));
//
//INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_const, ShapeOpsCPUTest, params_const, ShapeOpsCPUTest::getTestCaseName);
//
//} // namespace squeezeTest
//
//namespace unsqueezeTest {
//
//inferShapes noBounds{{-1, -1, -1, -1},
//                     {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{5, 1, 5}},
//                     {std::vector<int>{2, 5}, std::vector<int>{1, 2}, std::vector<int>{4, 5}, std::vector<int>{0, 1}}};
//
//const auto params = ::testing::Combine(::testing::Values(noBounds),
//                                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
//                                       ::testing::Values(shapeNodeType::Unsqueeze),
//                                       ::testing::Values(Precision::FP32),
//                                       ::testing::Values(true));
//
//// at this momemnt unsqueze produce dynamic output rank, if second input is not constant
//// enable after CPU plug-in will support dynamic rank
//// INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, ShapeOpsCPUTest, params, ShapeOpsCPUTest::getTestCaseName);
//
//inferShapes noBounds_const{{{1, 10}, {1, 15}, {2, 20}, {3, 7}},
//                           {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{1, 2, 5, 5}},
//                           {std::vector<int>{1, 3}}};
//
//const auto params_const = ::testing::Combine(::testing::Values(noBounds_const),
//                                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
//                                             ::testing::Values(shapeNodeType::Unsqueeze),
//                                             ::testing::Values(Precision::FP32),
//                                             ::testing::Values(true));
//
//INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_const, ShapeOpsCPUTest, params_const, ShapeOpsCPUTest::getTestCaseName);
//
//} // namespace unsqueezeTest
//
//} // namespace CPULayerTestsDefinitions
