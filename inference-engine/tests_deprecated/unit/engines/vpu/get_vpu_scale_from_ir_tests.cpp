// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"
#include "tests_vpu_common.hpp"

#include <cpp/ie_executable_network.hpp>

#include <ngraph/type/element_type.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <limits>

using namespace vpu;
using namespace InferenceEngine;

using VPU_AddVpuScaleTest = GraphTransformerTest;

TEST_F(VPU_AddVpuScaleTest, CanAddVpuScaleToNetwork) {
    InitCompileEnv();

    auto& env = CompileEnv::get();
    auto config = createConfiguration();
    config.set(InferenceEngine::MYRIAD_IR_WITH_SCALES_DIRECTORY, "/");
    env.updateConfig(config);

    std::shared_ptr<ngraph::Function> function;

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{4, 2, 2});
        input->set_friendly_name("input");
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{2, 2}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{2}, {1});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(input, weights, bias, ngraph::Shape{4, 2, 2});
        fc->set_friendly_name("FullyConnected");
        auto result = std::make_shared<ngraph::op::Result>(fc);
        ngraph::ResultVector results { result };
        ngraph::ParameterVector params {input };
        function = std::make_shared<ngraph::Function>(results, params);
    }

    auto network = InferenceEngine::CNNNetwork(function);
    auto model = frontEnd->buildInitialModel(network);

    const auto getFullyConnectedStage = [model]() -> Stage {
        const auto isFullyConnected = [](const Stage& stage) {
            const auto& layer = stage->origLayer();
            return layer && layer->type == "FullyConnected";
        };
        const auto stages = model->getStages();
        const auto stageIt = std::find_if(begin(stages), end(stages), isFullyConnected);
        return *stageIt;
    };

    const auto fcStage = getFullyConnectedStage();
    EXPECT_EQ(fcStage->origLayer()->params.find("vpu_scale"), fcStage->origLayer()->params.end());

    auto middleEnd = passManager->buildMiddleEnd();
    middleEnd->run(model);

    const auto fcStageAfterMiddleEnd = getFullyConnectedStage();
    EXPECT_NE(fcStageAfterMiddleEnd->origLayer()->params.find("vpu_scale"), fcStageAfterMiddleEnd->origLayer()->params.end());
}

TEST_F(VPU_AddVpuScaleTest, VpuScaleFromIrChangesWeights) {
    InitCompileEnv();
    const auto& env = CompileEnv::get();
    auto config = createConfiguration();
    config.set(InferenceEngine::MYRIAD_IR_WITH_SCALES_DIRECTORY, "/");
    env.updateConfig(config);

    std::shared_ptr<ngraph::Function> function;
    {
        ngraph::element::Type elementType = ngraph::element::Type_t::f16;
        ngraph::Shape shape { 1, 1, 4, 5 };
        auto input = std::make_shared<ngraph::op::Parameter>(elementType, shape);
        input->set_friendly_name("input");

        auto weights = std::make_shared<ngraph::op::Constant>(
                elementType, ngraph::Shape{1, 1, 1, 1}, std::vector<float>(1, 1.0f));
        auto conv = std::make_shared<ngraph::op::v1::Convolution>(
                input, weights, ngraph::Strides {1, 1},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{1, 1});
        conv->set_friendly_name("Convolution");
        auto result = std::make_shared<ngraph::op::Result>(conv);

        ngraph::ResultVector results { result };
        ngraph::ParameterVector params { input };
        function = std::make_shared<ngraph::Function>(results, params);
    }

    auto network = InferenceEngine::CNNNetwork(function);
    auto model = frontEnd->buildInitialModel(network);

    auto middleEnd = passManager->buildMiddleEnd();
    auto checkWeightWasChanged = [this, &network](const float scale) {
        auto model = frontEnd->buildInitialModel(network);
        for (const auto& stage : model->getStages()) {
            if (stage->name() == "Convolution") {
                stage->origLayer()->params["vpu_scale"] = toString(scale);
            }
        }

        auto middleEnd = passManager->buildMiddleEnd();
        middleEnd->run(model);
        for (const auto& stage : model->getStages()) {
            if (stage->name() == "Convolution") {
                auto content = stage->input(1)->content()->get<ie_fp16>();
                if (scale < 0) {
                    EXPECT_EQ(scale, PrecisionUtils::f16tof32(content[0]));
                } else {
                    EXPECT_EQ(scale, fabs(PrecisionUtils::f16tof32(content[0])));
                }
            }
        }
    };

    const auto maxVal = std::numeric_limits<float>::infinity();

    checkWeightWasChanged(32);
    checkWeightWasChanged(64);
    checkWeightWasChanged(maxVal);

}
