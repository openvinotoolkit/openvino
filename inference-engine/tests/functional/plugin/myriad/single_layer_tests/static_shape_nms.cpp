// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <functional_test_utils/blob_utils.hpp>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/op/non_max_suppression.hpp>
#include "vpu/ngraph/operations/static_shape_non_maximum_suppression.hpp"

using TensorShape = InferenceEngine::SizeVector;

using StaticShapeNMSParam = std::tuple<
        int64_t, // Number of batches
        int64_t, // Number of boxes
        int64_t, // Number of classes
        int64_t, // Maximum output boxes per class
        float, // IOU threshold
        float,  // Score threshold
        float>; // Soft NMS sigma

using StaticShapeNMSTestParam = std::tuple<
        StaticShapeNMSParam,             // NMS params
        InferenceEngine::Precision,      // Input precision
        LayerTestsUtils::TargetDevice>;  // Device name

namespace LayerTestsDefinitions {

class StaticShapeNMSLayerTest : public testing::WithParamInterface<StaticShapeNMSTestParam>,
                                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeNMSTestParam>& obj) {
        StaticShapeNMSParam NMSParams;
        InferenceEngine::Precision inputPrecision;
        std::string targetDevice;
        std::tie(NMSParams, inputPrecision, targetDevice) = obj.param;

        const auto numBatches = std::get<0>(NMSParams);
        const auto numBoxes = std::get<1>(NMSParams);
        const auto numClasses = std::get<2>(NMSParams);
        const auto maxOutputBoxesPerClass = std::get<3>(NMSParams);
        const auto iouThreshold = std::get<4>(NMSParams);
        const auto scoreThreshold = std::get<5>(NMSParams);

        std::ostringstream result;
        result << "numBatches=" << numBatches << "_";
        result << "numBoxes=" << numBoxes << "_";
        result << "numClasses=" << numClasses << "_";
        result << "maxOutputBoxesPerClass=" << maxOutputBoxesPerClass << "_";
        result << "iouThreshold=" << iouThreshold << "_";
        result << "scoreThreshold=" << scoreThreshold << "_";
        result << "inPrc=" << inputPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        StaticShapeNMSParam NMSParams;
        std::tie(NMSParams, inPrc, targetDevice) = this->GetParam();

        const auto numBatches = std::get<0>(NMSParams);
        const auto numBoxes = std::get<1>(NMSParams);
        const auto numClasses = std::get<2>(NMSParams);
        const auto maxOutputBoxesPerClass = std::get<3>(NMSParams);
        const auto iouThreshold = std::get<4>(NMSParams);
        const auto scoreThreshold = std::get<5>(NMSParams);
        const auto softNMSSigma = std::get<6>(NMSParams);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        const auto inputBoxes = std::make_shared<ngraph::opset3::Parameter>(
                ngPrc, ngraph::Shape({static_cast<size_t>(numBatches), static_cast<size_t>(numBoxes), 4}));
        const auto inputScores = std::make_shared<ngraph::opset3::Parameter>(
                ngPrc, ngraph::Shape({static_cast<size_t>(numBatches), static_cast<size_t>(numClasses), static_cast<size_t>(numBoxes)}));
        const auto maxOutputBoxesPerClassConst = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{}, maxOutputBoxesPerClass);
        const auto iouThresholdConst = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, ngraph::Shape{}, iouThreshold);
        const auto scoreThresholdConst = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, ngraph::Shape{}, scoreThreshold);
        const auto softNMSSigmaConst = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, ngraph::Shape{1}, softNMSSigma);

        const auto staticShapeNMS = std::make_shared<ngraph::vpu::op::StaticShapeNonMaxSuppression>(
                inputBoxes, inputScores, maxOutputBoxesPerClassConst, iouThresholdConst, scoreThresholdConst, softNMSSigmaConst,
                0, false, ngraph::element::i32);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(staticShapeNMS->output(0)),
                                     std::make_shared<ngraph::opset3::Result>(staticShapeNMS->output(1))};
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{inputBoxes, inputScores});
    }
};

TEST_P(StaticShapeNMSLayerTest, accuracy) {
    Run();
}

std::vector<StaticShapeNMSParam> NMSParams = {
        std::make_tuple(1, 10, 5, 10, 0., 0., 0.),
        std::make_tuple(2, 100, 5, 10, 0., 0., 0.),
        std::make_tuple(3, 10, 5, 2, 0.5, 0., 0.),
        std::make_tuple(1, 1000, 1, 2000, 0.5, 0., 0.),
        std::make_tuple(1, 8200, 1, 8200, 0.5, 0., 0.),
};

std::vector<InferenceEngine::Precision> NMSPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

// #-30919
INSTANTIATE_TEST_CASE_P(DISABLED_accuracy, StaticShapeNMSLayerTest,
        ::testing::Combine(
        ::testing::ValuesIn(NMSParams),
        ::testing::ValuesIn(NMSPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
        StaticShapeNMSLayerTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
