// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"

#include "vpu/private_plugin_config.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <tuple>
#include <vector>
#include <string>
#include <memory>

using TensorShape = InferenceEngine::SizeVector;

using StaticShapeBroadcastParam = std::tuple<
        TensorShape,   // Input shape
        TensorShape,   // Target shape
        TensorShape>;  // Axes mapping

using StaticShapeBroadcastTestParam = std::tuple<
        StaticShapeBroadcastParam,       // Shapes param
        InferenceEngine::Precision,      // Input precision
        LayerTestsUtils::TargetDevice>;  // Device name

namespace LayerTestsDefinitions {

class StaticShapeBroadcastLayerTest : public testing::WithParamInterface<StaticShapeBroadcastTestParam>,
                                      public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeBroadcastTestParam>& obj) {
        StaticShapeBroadcastParam shapes;
        InferenceEngine::Precision inputPrecision;
        std::string targetDevice;
        std::tie(shapes, inputPrecision, targetDevice) = obj.param;

        const auto inputShape = std::get<0>(shapes);
        const auto targetShape = std::get<1>(shapes);
        const auto axesMapping = std::get<2>(shapes);

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "TS=" << CommonTestUtils::vec2str(targetShape) << "_";
        if (!axesMapping.empty()) {
            result << "AM=" << CommonTestUtils::vec2str(axesMapping) << "_";
        }
        result << "inPrc=" << inputPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
        configuration[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

        StaticShapeBroadcastParam shapes;
        std::tie(shapes, inPrc, targetDevice) = this->GetParam();

        const auto inputShape = std::get<0>(shapes);
        const auto targetShape = std::get<1>(shapes);
        const auto axesMapping = std::get<2>(shapes);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        const auto inputParam = std::make_shared<ngraph::opset3::Parameter>(
                ngPrc, ngraph::Shape(inputShape));
        const auto targetShapeConst = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{targetShape.size()}, targetShape);

        std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> staticShapeBroadcast;
        if (axesMapping.empty()) {
            staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                    inputParam, targetShapeConst);
        } else {
            const auto axesMappingConst = std::make_shared<ngraph::opset3::Constant>(
                    ngraph::element::i64, ngraph::Shape{axesMapping.size()}, axesMapping);
            staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                    inputParam, targetShapeConst, axesMappingConst);
        }

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(staticShapeBroadcast->output(0))};
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{inputParam});
    }
};

TEST_P(StaticShapeBroadcastLayerTest, accuracy) {
    Run();
}

std::vector<StaticShapeBroadcastParam> broadcastParam = {
        std::make_tuple(TensorShape{ 14         }, TensorShape{  2, 16, 15, 14 }, TensorShape{}),
        std::make_tuple(TensorShape{ 15,  1     }, TensorShape{  2, 16, 15, 14 }, TensorShape{}),
        std::make_tuple(TensorShape{ 15, 14     }, TensorShape{  2, 16, 15, 14 }, TensorShape{}),
        std::make_tuple(TensorShape{ 16,  1,  1 }, TensorShape{  2, 16, 15, 14 }, TensorShape{}),
        std::make_tuple(TensorShape{ 16,  1, 14 }, TensorShape{  2, 16, 15, 14 }, TensorShape{}),
        std::make_tuple(TensorShape{ 16, 15,  1 }, TensorShape{  2, 16, 15, 14 }, TensorShape{}),

        std::make_tuple(TensorShape{ 80         }, TensorShape{ 80,  1         }, TensorShape{ 0 }),
        std::make_tuple(TensorShape{ 16         }, TensorShape{  1, 16, 50, 50 }, TensorShape{ 1 }),
        std::make_tuple(TensorShape{ 50, 50     }, TensorShape{  1, 50, 50, 16 }, TensorShape{ 1, 2 }),
};

std::vector<InferenceEngine::Precision> broadcastPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_CASE_P(accuracy, StaticShapeBroadcastLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(broadcastParam),
                                ::testing::ValuesIn(broadcastPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        StaticShapeBroadcastLayerTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
