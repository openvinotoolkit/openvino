// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include "single_layer_tests/activation.hpp"

namespace FusingTestUtils {

typedef std::tuple<
        std::shared_ptr<ngraph::Function>,
        std::vector<std::shared_ptr<ngraph::Node>>,
        std::vector<std::string>> fusingSpecificParams;

const std::vector<size_t> fakeShape = {1};
const auto fakeConstNode = ngraph::builder::makeConstant(ngraph::element::f32, fakeShape, {}, true);
const auto fakeParamNode = ngraph::builder::makeParams(ngraph::element::f32, {fakeShape})[0];

/* FUSING PATTERNS */
const auto fusingRelu = fusingSpecificParams{nullptr,
        {ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::Relu)}, {"Relu"}};
const auto fusingElu = fusingSpecificParams{nullptr,
        {ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::Elu, 2.0f)}, {"Elu"}};
const auto fusingSigmoid = fusingSpecificParams{nullptr,
        {ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::Sigmoid)}, {"Sigmoid"}};
const auto fusingClamp = fusingSpecificParams{nullptr,
        {ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::Clamp, 3.0f, 6.0f)}, {"Clamp"}};
const auto fusingPRelu = fusingSpecificParams{nullptr,
        {fakeConstNode, ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::LeakyRelu)}, {"PRelu"}};

const auto fusingReluScaleShift = fusingSpecificParams{nullptr,
        {ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::Relu),
         fakeConstNode, std::make_shared<ngraph::opset1::Multiply>(fakeParamNode, fakeConstNode),
         fakeConstNode, std::make_shared<ngraph::opset1::Add>(fakeParamNode, fakeConstNode)}, {"Relu", "Add"}};
 const auto fusingFakeQuantizeRelu = fusingSpecificParams{nullptr,
        {fakeConstNode, fakeConstNode, fakeConstNode, fakeConstNode,
         ngraph::builder::makeFakeQuantize(fakeParamNode, ngraph::element::f32, 256, fakeShape),
         ngraph::builder::makeActivation(fakeParamNode, ngraph::element::f32, ngraph::helpers::Relu)}, {"FakeQuantize", "Relu"}};
const auto fusingSum = fusingSpecificParams{nullptr,
        {fakeParamNode, std::make_shared<ngraph::opset1::Add>(fakeParamNode, fakeParamNode)}, {"Add"}};
// todo: DWConvolution (not supported for GroupConvolution)


/* "HARD" FUSING PATTERNS */
std::shared_ptr<ngraph::Function> makeSwishPattern(std::vector<size_t> shape);
std::shared_ptr<ngraph::Function> makeFakeQuantizeActivationPattern(size_t levels, ngraph::helpers::ActivationTypes type, std::vector<size_t> shape);


std::string postNodes2str(const std::vector<std::shared_ptr<ngraph::Node>> &postNodes);
std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
        const std::shared_ptr<ngraph::Node> &lastNode, const std::shared_ptr<ngraph::Function> &postFunction);
std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
        const std::shared_ptr<ngraph::Node> &lastNode, const std::vector<std::shared_ptr<ngraph::Node>> &postNodes);


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
