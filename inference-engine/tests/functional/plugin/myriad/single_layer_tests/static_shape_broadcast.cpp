// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"

#include "vpu/private_plugin_config.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
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
        TensorShape,   // Axes mapping
        std::string>;  // mode

using StaticShapeBroadcastTestParam = std::tuple<
        StaticShapeBroadcastParam,       // Shapes param
        InferenceEngine::Precision,      // Input precision
        LayerTestsUtils::TargetDevice>;  // Device name

namespace LayerTestsDefinitions {

class StaticShapeBroadcastLayerTest : public testing::WithParamInterface<StaticShapeBroadcastTestParam>,
                                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeBroadcastTestParam>& obj) {
        StaticShapeBroadcastParam shapes;
        InferenceEngine::Precision inputPrecision;
        std::string targetDevice;
        std::tie(shapes, inputPrecision, targetDevice) = obj.param;

        const auto inputShape = std::get<0>(shapes);
        const auto targetShape = std::get<1>(shapes);
        const auto axesMapping = std::get<2>(shapes);
        const auto mode = std::get<3>(shapes);

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "TS=" << CommonTestUtils::vec2str(targetShape) << "_";
        result << "mode=" << mode << "_";
        if (mode == "explicit") {
            result << "AM=" << CommonTestUtils::vec2str(axesMapping) << "_";
        }
        result << "inPrc=" << inputPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        StaticShapeBroadcastParam shapes;
        std::tie(shapes, inPrc, targetDevice) = this->GetParam();

        const auto inputShape = std::get<0>(shapes);
        const auto targetShape = std::get<1>(shapes);
        const auto axesMapping = std::get<2>(shapes);
        const auto mode = std::get<3>(shapes);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        const auto inputParam = std::make_shared<ngraph::opset3::Parameter>(
                ngPrc, ngraph::Shape(inputShape));
        const auto targetShapeConst = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{targetShape.size()}, targetShape);

        std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> staticShapeBroadcast;
        if (mode == "numpy") {
            staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                    inputParam, targetShapeConst);
        } else if (mode == "explicit") {
            const auto axesMappingConst = std::make_shared<ngraph::opset3::Constant>(
                    ngraph::element::i64, ngraph::Shape{axesMapping.size()}, axesMapping);
            staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                    inputParam, targetShapeConst, axesMappingConst);
        } else {
            staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                    inputParam, targetShapeConst, ngraph::op::BroadcastType::BIDIRECTIONAL);
        }

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(staticShapeBroadcast->output(0))};
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{inputParam});
    }
};

TEST_P(StaticShapeBroadcastLayerTest, accuracy) {
    Run();
}

std::vector<StaticShapeBroadcastParam> broadcastParam = {
        std::make_tuple(TensorShape{ 14         }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "numpy"),
        std::make_tuple(TensorShape{ 15,  1     }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "numpy"),
        std::make_tuple(TensorShape{ 15, 14     }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "numpy"),
        std::make_tuple(TensorShape{ 16,  1,  1 }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "numpy"),
        std::make_tuple(TensorShape{ 16,  1, 14 }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "numpy"),
        std::make_tuple(TensorShape{ 16, 15,  1 }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "numpy"),

        std::make_tuple(TensorShape{ 80         }, TensorShape{ 80,  1         }, TensorShape{ 0 }, "explicit"),
        std::make_tuple(TensorShape{ 16         }, TensorShape{  1, 16, 50, 50 }, TensorShape{ 1 }, "explicit"),
        std::make_tuple(TensorShape{ 50, 50     }, TensorShape{  1, 50, 50, 16 }, TensorShape{ 1, 2 }, "explicit"),

        std::make_tuple(TensorShape{ 14         }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "bidirectional"),
        std::make_tuple(TensorShape{ 15,  1     }, TensorShape{  2, 16, 15, 14 }, TensorShape{}, "bidirectional"),
        std::make_tuple(TensorShape{  2, 16, 15, 14 }, TensorShape{ 15, 14     }, TensorShape{}, "bidirectional"),
        std::make_tuple(TensorShape{  2, 16, 15, 14 }, TensorShape{ 16,  1,  1 }, TensorShape{}, "bidirectional"),
        std::make_tuple(TensorShape{  2, 16, 15, 14 }, TensorShape{ 16,  1, 14 }, TensorShape{}, "bidirectional"),
        std::make_tuple(TensorShape{ 16, 15,  1 }, TensorShape{  2, 1, 15, 14  }, TensorShape{}, "bidirectional"),
};

std::vector<InferenceEngine::Precision> broadcastPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_SUITE_P(smoke_accuracy, StaticShapeBroadcastLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(broadcastParam),
                                ::testing::ValuesIn(broadcastPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        StaticShapeBroadcastLayerTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
