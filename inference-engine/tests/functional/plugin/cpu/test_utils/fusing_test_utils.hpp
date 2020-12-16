// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include <shared_test_classes/single_layer/activation.hpp>

namespace CPUTestUtils {

struct postNode {
    std::function<std::shared_ptr<ngraph::Node>(std::shared_ptr<ngraph::Node>, const ngraph::element::Type&, ngraph::ParameterVector&)> makeNode;
    std::string name;
};

typedef std::tuple<
        std::shared_ptr<ngraph::Function>,
        std::vector<postNode>,
        std::vector<std::string>> fusingSpecificParams;

class CpuTestWithFusing : public CPUTestsBase {
public:
    static std::string getTestCaseName(fusingSpecificParams params);

protected:
    std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                                      ngraph::ParameterVector &params,
                                                      const std::shared_ptr<ngraph::Node> &lastNode) const override;

    void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const override;

private:
    std::shared_ptr<ngraph::Node> addPostFunction(const std::shared_ptr<ngraph::Node> &lastNode) const;
    std::shared_ptr<ngraph::Node> addPostNodes(const ngraph::element::Type &ngPrc,
                                               ngraph::ParameterVector &params,
                                               const std::shared_ptr<ngraph::Node> &lastNode) const;
protected:
    std::shared_ptr<ngraph::Function> postFunction;
    std::vector<postNode> postNodes;
    std::vector<std::string> fusedOps;
};

/* FUSING PATTERNS */
const auto emptyFusingSpec = fusingSpecificParams{nullptr, {}, {}};
const auto fusingRelu = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"}}, {"Relu"}};
const auto fusingElu = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Elu, {}, {2.0f});
            }, "Elu"}}, {"Elu"}};
const auto fusingSigmoid = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Sigmoid);
            }, "Sigmoid"}}, {"Sigmoid"}};
const auto fusingClamp = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
            }, "Clamp"}}, {"Clamp"}};
const auto fusingPRelu = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                    THROW_IE_EXCEPTION << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(newShape));
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::LeakyRelu, newShape, data);
            }, "PRelu(PerChannel)"}}, {"PRelu"}};
const auto fusingSwish = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Swish, {}, {1.0f});
            }, "Swish"}}, {"Swish"}};

const auto fusingReluScaleShift = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                 auto shape = inpNode->get_shape();
                 if (shape.size() == 1)
                     THROW_IE_EXCEPTION << "If shape.size() == 1 then Granularity can be PerTensor only";
                 ngraph::Shape newShape(shape.size(), 1);
                 newShape[1] = shape[1];
                 auto constNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, newShape, {}, true);
                 return std::make_shared<ngraph::opset1::Multiply>(inpNode, constNode);
            }, "Multiply(PerChannel)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                 THROW_IE_EXCEPTION << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                auto constNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, newShape, {}, true);
                return std::make_shared<ngraph::opset1::Add>(inpNode, constNode);
            }, "Add(PerChannel)"}}, {"Relu", "Add"}};
const auto fusingFakeQuantizePerChannelRelu = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto localPrc = ngPrc;
                if (ngraph::element::bf16 == localPrc) {
                    localPrc = ngraph::element::f32;
                }
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                 THROW_IE_EXCEPTION << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"}}, {"FakeQuantize", "Relu"}};
const auto fusingFakeQuantizePerTensorRelu = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
                auto localPrc = ngPrc;
                if (ngraph::element::bf16 == localPrc) {
                    localPrc = ngraph::element::f32;
                }
                auto newShape = ngraph::Shape(inpNode->get_shape().size(), 1);
                return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"}}, {"FakeQuantize", "Relu"}};
const auto fusingSum = fusingSpecificParams{nullptr, {
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                ngraph::ParameterVector newParams = ngraph::builder::makeParams(ngPrc, {shape});
                params.insert(params.end(), newParams.begin(), newParams.end());
                auto newParamOuts = ngraph::helpers::convert2OutputVector(
                     ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(newParams));
                return std::make_shared<ngraph::opset1::Add>(inpNode, newParamOuts[0]);
            }, "Add(Parameters)"}}, {"Add"}};
// todo: DWConvolution (not supported for GroupConvolution)

} // namespace CPUTestUtils
