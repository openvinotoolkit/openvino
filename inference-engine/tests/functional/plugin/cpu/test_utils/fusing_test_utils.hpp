// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include <shared_test_classes/single_layer/activation.hpp>

namespace CPUTestUtils {

struct postNodeBuilder {
    std::function<std::shared_ptr<ngraph::Node>(std::shared_ptr<ngraph::Node>, const ngraph::element::Type&, ngraph::ParameterVector&)> makeNode;
    std::string name;
};

class postOpMgr {
public:
    virtual std::shared_ptr<ngraph::Node> addPostOps(const ngraph::element::Type &ngPrc,
                                                     ngraph::ParameterVector &params,
                                                     const std::shared_ptr<ngraph::Node> &lastNode) const = 0;
    virtual std::string getFusedOpsNames() const = 0;
    virtual ~postOpMgr() = default;
};

class postFunctionMgr : public postOpMgr {
public:
    postFunctionMgr(std::shared_ptr<ngraph::Function> function) : _pFunction(function) {}
    std::shared_ptr<ngraph::Node> addPostOps(const ngraph::element::Type &ngPrc,
                                             ngraph::ParameterVector &params,
                                             const std::shared_ptr<ngraph::Node> &lastNode) const override;
    std::string getFusedOpsNames() const override;

private:
    std::shared_ptr<ngraph::Function> _pFunction;
};

class postNodesMgr : public postOpMgr {
public:
    postNodesMgr(std::vector<postNodeBuilder> postNodes);
    std::shared_ptr<ngraph::Node> addPostOps(const ngraph::element::Type &ngPrc,
                                             ngraph::ParameterVector &params,
                                             const std::shared_ptr<ngraph::Node> &lastNode) const override;
    std::string getFusedOpsNames() const override;

private:
    std::vector<postNodeBuilder> _postNodes;
};

typedef std::tuple<
        std::shared_ptr<postOpMgr>, // post operation manager (add post operations to the graph)
        std::vector<std::string> // list of node types that are to be fused
        > fusingSpecificParams;

class CpuTestWithFusing : public CPUTestsBase {
public:
    static std::string getTestCaseName(fusingSpecificParams params);

protected:
    /**
     * @brief This function adds post operations.
     */
    std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                              ngraph::ParameterVector &params,
                                              const std::shared_ptr<ngraph::Node> &lastNode) const override;

    void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const override;
    void CheckFusingResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const;

protected:
    std::shared_ptr<postOpMgr> postOpMgrPtr;
    std::vector<std::string> fusedOps;
    bool checkFusingPosition = true;
};

/* FUSING PATTERNS */
const auto emptyFusingSpec = fusingSpecificParams{nullptr, {}};

const auto fusingRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"}}), {"Relu"}};

const auto fusingElu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Elu, {}, {2.0f});
            }, "Elu"}}), {"Elu"}};

const auto fusingGelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Gelu);
            }, "Gelu"}}), {"Gelu"}};

const auto fusingSigmoid = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Sigmoid);
            }, "Sigmoid"}}), {"Sigmoid"}};

const auto fusingClamp = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
            }, "Clamp"}}), {"Clamp"}};

const auto fusingTanh = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Tanh);
            }, "Tanh"}}), {"Tanh"}};

const auto fusingAbs = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Abs);
            }, "Abs"}}), {"Abs"}};

const auto fusingSqrt = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Sqrt);
            }, "Sqrt"}}), {"Sqrt"}};

const auto fusingPReluPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                    IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(newShape));
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::LeakyRelu, newShape, data);
            }, "PRelu(PerChannel)"}}), {"PRelu"}};

const auto fusingPReluPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                ngraph::Shape shape(1, 1);
                auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape));
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::LeakyRelu, shape, data);
            }, "PRelu(PerTensor)"}}), {"PRelu"}};

const auto fusingSwish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Swish, {}, {1.0f});
            }, "Swish"}}), {"Swish"}};

const auto fusingSoftPlus = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::SoftPlus, {}, {});
            }, "SoftPlus"}}), {"SoftPlus"}};

const auto fusingHSwish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::HSwish, {}, {});
            }, "HSwish"}}), {"HSwish"}};

const auto fusingMish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Mish, {}, {});
            }, "Mish"}}), {"Mish"}};

const auto fusingHSigmoid = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::HSigmoid);
            }, "HSigmoid"}}), {"HSigmoid"}};

const auto fusingReluAdd = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                    IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(inpNode, constNode);
            }, "Add(PerChannel)"}}), {"Relu", "Add"}};

const auto fusingReluScaleShift = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                 auto shape = inpNode->get_shape();
                 if (shape.size() == 1)
                     IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                 ngraph::Shape newShape(shape.size(), 1);
                 newShape[1] = shape[1];
                 auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
                 return std::make_shared<ngraph::opset1::Multiply>(inpNode, constNode);
            }, "Multiply(PerChannel)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                 IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(inpNode, constNode);
            }, "Add(PerChannel)"}}), {"Relu", "Add"}};

const auto fusingScaleShift = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
                 auto shape = inpNode->get_shape();
                 if (shape.size() == 1)
                     IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                 ngraph::Shape newShape(shape.size(), 1);
                 newShape[1] = shape[1];
                 auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
                 return std::make_shared<ngraph::opset1::Multiply>(inpNode, constNode);
            }, "Multiply(PerChannel)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                 IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(inpNode, constNode);
            }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingFakeQuantizePerTensor = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto localPrc = inpNode->get_element_type();
                ngraph::Shape newShape(inpNode->get_shape().size(), 1);
                return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"}}), {"FakeQuantize"} };

const auto fusingFakeQuantizePerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto localPrc = inpNode->get_element_type();
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                    IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizePerChannelRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto localPrc = inpNode->get_element_type();
                auto shape = inpNode->get_shape();
                if (shape.size() == 1)
                 IE_THROW() << "If shape.size() == 1 then Granularity can be PerTensor only";
                ngraph::Shape newShape(shape.size(), 1);
                newShape[1] = shape[1];
                return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"}}), {"FakeQuantize", "Relu"}};

const auto fusingFakeQuantizePerTensorRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
                auto localPrc = inpNode->get_element_type();
                auto newShape = ngraph::Shape(inpNode->get_shape().size(), 1);
                return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"},
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Relu);
            }, "Relu"}}), {"FakeQuantize", "Relu"}};

const auto fusingSum = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
                auto shape = inpNode->get_shape();
                ngraph::ParameterVector newParams = ngraph::builder::makeParams(ngPrc, {shape});
                params.insert(params.end(), newParams.begin(), newParams.end());
                auto newParamOuts = ngraph::helpers::convert2OutputVector(
                     ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(newParams));
                return std::make_shared<ngraph::opset1::Add>(inpNode, newParamOuts[0]);
            }, "Add(Parameters)"}}), {"Add"}};

const auto fusingSumEluFQ = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            auto shape = inpNode->get_shape();
            ngraph::ParameterVector newParams = ngraph::builder::makeParams(ngPrc, {shape});
            params.insert(params.end(), newParams.begin(), newParams.end());
            auto newParamOuts = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(newParams));
            return std::make_shared<ngraph::opset1::Add>(inpNode, newParamOuts[0]);
        }, "Add(Parameters)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Elu, {}, {2.0f});
        }, "Elu"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
            auto localPrc = inpNode->get_element_type();
            auto newShape = ngraph::Shape(inpNode->get_shape().size(), 1);
            return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
        }, "FakeQuantize(PerTensor)"}}), {"Add", "Elu", "FakeQuantize"}};

const auto fusingMultiplyPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::op::v1::Multiply>(inpNode, secondMultInput);
        }, "Multiply(PerTensor)"}}), {"Multiply"}};

const auto fusingMultiplyPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(inpNode->get_shape().size(), 1);
            secondMultInShape[1] = inpNode->get_shape()[1];
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(inpNode, secondMultInput);
        }, "Multiply(PerChannel)"}}), {"Multiply"}};

const auto fusingAddPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(inpNode, secondMultInput);
        }, "Add(PerTensor)"}}), {"Add"}};

const auto fusingAddPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(inpNode->get_shape().size(), 1);
            secondMultInShape[1] = inpNode->get_shape()[1];
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(inpNode, secondMultInput);
        }, "Add(PerChannel)"}}), {"Add"}};

const auto fusingSubtractPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(inpNode, secondMultInput);
        }, "Subtract(PerTensor)"}}), {"Subtract"}};

const auto fusingSubtractPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(inpNode->get_shape().size(), 1);
            secondMultInShape[1] = inpNode->get_shape()[1];
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(inpNode, secondMultInput);
        }, "Subtract(PerChannel)"}}), {"Subtract"}};

const auto fusingDividePerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(inpNode, secondMultInput);
        }, "Divide(PerTensor)"}}), {"Divide"}};

const auto fusingDividePerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape(inpNode->get_shape().size(), 1);
            secondMultInShape[1] = inpNode->get_shape()[1];
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, ngraph::Shape(secondMultInShape), std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(inpNode, secondMultInput);
        }, "Divide(PerChannel)"}}), {"Divide"}};

const auto fusingPRelu1D = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
    {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
        auto shape = inpNode->get_shape();
        ngraph::Shape newShape({shape[1]});
        auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(newShape));
        return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::LeakyRelu, newShape, data);
    }, "PRelu1D"}}), {"PRelu"}};

} // namespace CPUTestUtils
