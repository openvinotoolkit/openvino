// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include "single_layer_tests/activation.hpp"

namespace FusingTestUtils {

struct postNode {
    std::shared_ptr<ngraph::Node> nodePtr;
    std::map<std::string, std::string> addInfo;
};

typedef std::tuple<
        std::shared_ptr<ngraph::Function>,
        std::vector<postNode>,
        std::vector<std::string>> fusingSpecificParams;

const std::vector<size_t> fakeShape = {1};
const auto fakeConstNode = ngraph::builder::makeConstant(ngraph::element::f32, fakeShape, {}, true);

/* FUSING PATTERNS */
const auto fusingRelu = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Relu)}}, {"Relu"}};
const auto fusingElu = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Elu, 2.0f)}}, {"Elu"}};
const auto fusingSigmoid = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Sigmoid)}}, {"Sigmoid"}};
const auto fusingClamp = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Clamp, 3.0f, 6.0f)}}, {"Clamp"}};
const auto fusingPRelu = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::LeakyRelu), {{"Granularity" , "PerChannel"}}}}, {"PRelu"}};

const auto fusingReluScaleShift = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Relu)},
         {std::make_shared<ngraph::opset1::Multiply>(fakeConstNode, fakeConstNode), {{"Granularity" , "PerChannel"}}},
         {std::make_shared<ngraph::opset1::Add>(fakeConstNode, fakeConstNode), {{"Granularity" , "PerChannel"}}}}, {"Relu", "Add"}};
const auto fusingFakeQuantizePerChannelRelu = fusingSpecificParams{nullptr,
        {{ngraph::builder::makeFakeQuantize(fakeConstNode, ngraph::element::f32, 256, fakeShape), {{"Granularity", "PerChannel"}}},
         {ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Relu)}}, {"FakeQuantize", "Relu"}};
// TODO: failed test
// const auto fusingFakeQuantizePerTensorRelu = fusingSpecificParams{nullptr,
//         {{ngraph::builder::makeFakeQuantize(fakeConstNode, ngraph::element::f32, 256, fakeShape), {{"Granularity", "PerTensor"}}},
//          {ngraph::builder::makeActivation(fakeConstNode, ngraph::element::f32, ngraph::helpers::Relu)}}, {"FakeQuantize", "Relu"}};
const auto fusingSum = fusingSpecificParams{nullptr,
        {{std::make_shared<ngraph::opset1::Add>(fakeConstNode, fakeConstNode), {{"Inputs", "Parameters"}}}}, {"Add"}};
// todo: DWConvolution (not supported for GroupConvolution)

/* "HARD" FUSING PATTERNS */
std::shared_ptr<ngraph::Function> makeSwishPattern();
const auto fusingSwishPattern = fusingSpecificParams{makeSwishPattern(), {}, {"Swish"}};

std::string postNodes2str(const std::vector<postNode> &postNodes);
std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
        const std::shared_ptr<ngraph::Node> &lastNode, const std::shared_ptr<ngraph::Function> &postFunction);
std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
        const std::shared_ptr<ngraph::Node> &lastNode, const std::vector<postNode> &postNodes);

void inline CheckFusing(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, std::vector<std::string> fusedOps) {
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    for (const auto & op : function->get_ops()) {
        const auto & rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        auto layerType = getExecValue("layerType");
        if (layerType == nodeType) {
            auto originalLayersNames = getExecValue("originalLayersNames");
            auto pos = originalLayersNames.find(nodeType);
            ASSERT_TRUE(pos != std::string::npos);
            for (auto fusedOp : fusedOps) {
                pos = originalLayersNames.find(fusedOp, pos);
                ASSERT_TRUE(pos != std::string::npos);
            }
        }
    }
}

} // namespace FusingTestUtils
