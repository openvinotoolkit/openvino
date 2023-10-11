//// Copyright (C) 2018-2023 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "test_utils/cpu_test_utils.hpp"
//
//#include "ov_models/builders.hpp"
//#include "ov_models/utils/ov_helpers.hpp"
//
//using namespace InferenceEngine;
//using namespace CPUTestUtils;
//
//namespace CPULayerTestsDefinitions {
//typedef std::tuple<
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>,  // input shape
//        std::tuple<float, float, float>,  // start, limit, delta
//        Precision  // output type
//> RangeSpecificParams;
//
//typedef std::tuple<
//        RangeSpecificParams,
//        InferenceEngine::Precision,     // Net precision
//        LayerTestsUtils::TargetDevice   // Device name
//> RangeLayerTestParams;
//
//typedef std::tuple<
//        CPULayerTestsDefinitions::RangeLayerTestParams,
//        CPUSpecificParams> RangeLayerCPUTestParamsSet;
//
//class RangeLayerCPUTest : public testing::WithParamInterface<RangeLayerCPUTestParamsSet>,
//                             virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
//    float start = 0;
//    float stop = 0;
//    float step = 0;
//public:
//    static std::string getTestCaseName(testing::TestParamInfo<RangeLayerCPUTestParamsSet> obj) {
//        CPULayerTestsDefinitions::RangeLayerTestParams basicParamsSet;
//        CPUSpecificParams cpuParams;
//        std::tie(basicParamsSet, cpuParams) = obj.param;
//        std::string td;
//        Precision netPrc = Precision::FP32;
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
//
//        RangeSpecificParams rangePar;
//        std::tie(rangePar, netPrc, td) = basicParamsSet;
//        std::tuple<float, float, float> rangeInputs;
//        InferenceEngine::Precision outPrc = Precision::FP32;
//        std::tie(shapes, rangeInputs, outPrc) = rangePar;
//        float start = std::get<0>(rangeInputs);
//        float stop = std::get<1>(rangeInputs);
//        float step = std::get<2>(rangeInputs);
//
//        std::ostringstream result;
//        result << "RangeTest_" << std::to_string(obj.index) << "_";
//        result << "NetPr_" << netPrc.name() << "_";
//        result << "OutPr_" << outPrc.name() << "_";
//        result << "Start_" << start << "_";
//        result << "Stop_" << stop << "_";
//        result << "Step_" << step << "_";
//        result << CPUTestsBase::getTestCaseName(cpuParams);
//        result << ov::test::utils::vec2str(shapes.second[0]) << "_";
//        return result.str();
//    }
//protected:
//    void GenerateInputs() override {
//        // for correct work of fill_data_random() method
//        size_t blobFillingRange = (inPrc == Precision::FP32 ? 0 : 1);
//        inputs.clear();
//        const auto& inputsInfo = executableNetwork.GetInputsInfo();
//        const auto& functionParams = function->get_parameters();
//        for (int i = 0; i < functionParams.size(); ++i) {
//            const float scalarVal = (i == 0 ? start : (i == 1 ? stop : step));
//            const auto& param = functionParams[i];
//            const auto infoIt = inputsInfo.find(param->get_friendly_name());
//            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
//            InferenceEngine::InputInfo::CPtr info = infoIt->second;
//            InferenceEngine::Blob::Ptr blob = nullptr;
//            if (!inputDynamicShapes.empty()) {
//                if (inputDynamicShapes[i].rank() != 0) {
//                    InferenceEngine::DataPtr dataNew(
//                            new InferenceEngine::Data(infoIt->first, info->getTensorDesc().getPrecision(),
//                                                      targetStaticShapes[index][i],
//                                                      info->getTensorDesc().getLayout()));
//                    InferenceEngine::InputInfo infoNew;
//                    infoNew.setInputData(dataNew);
//                    blob = FuncTestUtils::createAndFillBlob(infoNew.getTensorDesc(), blobFillingRange, scalarVal);
//                }
//            }
//            if (blob == nullptr) {
//                blob = FuncTestUtils::createAndFillBlob((*info).getTensorDesc(), blobFillingRange, scalarVal);
//            }
//            inputs.push_back(blob);
//        }
//    }
//
//    void SetUp() override {
//        CPULayerTestsDefinitions::RangeLayerTestParams basicParamsSet;
//        CPUSpecificParams cpuParams;
//        std::tie(basicParamsSet, cpuParams) = this->GetParam();
//        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
//        CPULayerTestsDefinitions::RangeSpecificParams rangeParams;
//        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
//        std::tie(rangeParams, inPrc, targetDevice) = basicParamsSet;
//        std::tuple<float, float, float> rangeInputs;
//
//        std::tie(shapes, rangeInputs, outPrc) = rangeParams;
//        targetStaticShapes = shapes.second;
//        inputDynamicShapes = shapes.first;
//
//        start = std::get<0>(rangeInputs);
//        stop = std::get<1>(rangeInputs);
//        step = std::get<2>(rangeInputs);
//        auto ngOutPr = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
//        auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
//        auto startPar = std::make_shared<ngraph::opset5::Parameter>(ngNetPrc, ngraph::Shape{});
//        auto stopPar = std::make_shared<ngraph::opset5::Parameter>(ngNetPrc, ngraph::Shape{});
//        auto stepPar = std::make_shared<ngraph::opset5::Parameter>(ngNetPrc, ngraph::Shape{});
//        auto range = std::make_shared<ngraph::opset4::Range>(startPar, stopPar, stepPar, ngOutPr);
//        range->get_rt_info() = getCPUInfo();
//        selectedType = std::string("ref_any_") + (inPrc == outPrc ? inPrc.name() : "FP32");
//        startPar->set_friendly_name("start");
//        stopPar->set_friendly_name("stop");
//        stepPar->set_friendly_name("step");
//
//        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(range)};
//        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector {
//            startPar, stopPar, stepPar}, "Range");
//        functionRefs = ngraph::clone_function(*function);
//    }
//};
//
//TEST_P(RangeLayerCPUTest, CompareWithRefs) {
//    run();
//    CheckPluginRelatedResults(executableNetwork, "Range");
//}
//
//namespace {
//
///* CPU PARAMS */
//std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
//    return std::vector<CPUSpecificParams> {CPUSpecificParams{{}, {x}, {}, {}}};
//}
//
//const std::vector<InferenceEngine::Precision> netPrecisions = {
//        InferenceEngine::Precision::FP32,
//        InferenceEngine::Precision::I32
//};
//const std::vector<InferenceEngine::Precision> outputType = {
//        InferenceEngine::Precision::FP32,
//        InferenceEngine::Precision::I32
//};
//
//std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic = {
//        {{ngraph::PartialShape(), ngraph::PartialShape(), ngraph::PartialShape()},
//         {{ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape{}}, {ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape{}}}}
//};
//std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesPseudoStatic = {
//        {{}, {{ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape{}}}}
//};
//
//const std::vector<std::tuple<float, float, float>> rangeInputValues = {
//        std::tuple<float, float, float> {1.0, -5.0, -1.0},
//        std::tuple<float, float, float> {1.0, 10.0, 1.2},
//        std::tuple<float, float, float> {1.1, 12.2, 1.1},
//        std::tuple<float, float, float> {1.1, -5.1, -1.1},
//        std::tuple<float, float, float> {1.0, 5.0, 2.0},
//        std::tuple<float, float, float> {10.0, 6.0, -3.0},
//        std::tuple<float, float, float> {5, 35, 5}
//};
//const auto rangeParDynamic = ::testing::Combine(
//        ::testing::ValuesIn(inShapesDynamic),
//        ::testing::ValuesIn(rangeInputValues),
//        ::testing::ValuesIn(outputType)
//);
//const auto rangeParStatic = ::testing::Combine(
//        ::testing::ValuesIn(inShapesPseudoStatic),
//        ::testing::ValuesIn(rangeInputValues),
//        ::testing::ValuesIn(outputType)
//);
//const auto params3dDynamic = ::testing::Combine(
//        ::testing::Combine(
//                rangeParDynamic,
//                ::testing::ValuesIn(netPrecisions),
//                ::testing::Values(ov::test::utils::DEVICE_CPU)),
//        ::testing::ValuesIn(filterCPUInfoForDevice()));
//const auto params3dPseudoStatic = ::testing::Combine(
//        ::testing::Combine(
//                rangeParStatic,
//                ::testing::ValuesIn(netPrecisions),
//                ::testing::Values(ov::test::utils::DEVICE_CPU)),
//        ::testing::ValuesIn(filterCPUInfoForDevice()));
//// We don't check static case, because of constant folding, but we can use static shape for test infrastructure,
//// however Range node will be dynamic, since inputs are parameters, not a constants
//INSTANTIATE_TEST_SUITE_P(smoke_RangePseudoStaticLayoutTest, RangeLayerCPUTest,
//                         params3dPseudoStatic, RangeLayerCPUTest::getTestCaseName);
//INSTANTIATE_TEST_SUITE_P(smoke_RangeDynamicLayoutTest, RangeLayerCPUTest,
//                         params3dDynamic, RangeLayerCPUTest::getTestCaseName);
//} // namespace
//} // namespace CPULayerTestsDefinitions
