// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include <memory>
#include <shared_test_classes/single_layer/activation.hpp>

namespace CPUTestUtils {

struct postNodeConfig {
    const std::shared_ptr<ngraph::Node> target;
    std::shared_ptr<ngraph::Node> input;
    const ngraph::element::Type& type;
    ngraph::ParameterVector& params;
};

struct postNodeBuilder {
    std::function<std::shared_ptr<ngraph::Node>(postNodeConfig& cfg)> makeNode;
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
    postFunctionMgr(std::shared_ptr<ngraph::Function> function) : _pFunction(std::move(function)) {}
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
                                              const std::shared_ptr<ngraph::Node> &lastNode) override;

    void CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const override;

private:
    void CheckFusingResults(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const;

protected:
    std::shared_ptr<postOpMgr> postOpMgrPtr;
    std::vector<std::string> fusedOps;
    bool checkFusingPosition = true;
};

static int getChannelAxis(const ov::AxisSet &axes, bool keep_dims) {
    int channelAxis = 1;
    if (!keep_dims) {
        for (auto axis : axes) {
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            } else if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}

static int getFusingAxis(const std::shared_ptr<ngraph::Node>& node) {
    if (std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(node)) {
        return node->get_output_partial_shape(0).size() - 1; // last dimension
    } else if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(node)) {
        return getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
    } else if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::LogicalReductionKeepDims>(node)) {
        return getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
    } else {
        return 1; // second dimension
    }
}

static ngraph::Shape generatePerChannelShape(const std::shared_ptr<ngraph::Node>& node) {
    const auto shape = node->get_output_partial_shape(0);
    if (shape.size() == 0)
        OPENVINO_THROW("If shape.size() == 0 then PerTensor fusing tests are N/A");
    if (shape.size() == 1)
        OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
    ngraph::Shape perChannelShape(shape.size(), 1);
    const auto channelAxis = getFusingAxis(node);
    if (channelAxis >= 0)
        perChannelShape[channelAxis] = shape[channelAxis].get_length();

    return perChannelShape;
}

/* FUSING PATTERNS */
const auto emptyFusingSpec = fusingSpecificParams{nullptr, {}};

const auto fusingRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Relu);
            }, "Relu"}}), {"Relu"}};

const auto fusingElu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Elu, {}, {2.0f});
            }, "Elu"}}), {"Elu"}};

const auto fusingGelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Gelu);
            }, "Gelu"}}), {"Gelu"}};

const auto fusingSigmoid = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Sigmoid);
            }, "Sigmoid"}}), {"Sigmoid"}};

const auto fusingClamp = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
            }, "Clamp"}}), {"Clamp"}};

const auto fusingTanh = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Tanh);
            }, "Tanh"}}), {"Tanh"}};

const auto fusingAbs = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Abs);
            }, "Abs"}}), {"Abs"}};

const auto fusingSqrt = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Sqrt);
            }, "Sqrt"}}), {"Sqrt"}};

const auto fusingPReluPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(newShape));
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::LeakyRelu, newShape, data);
            }, "PRelu(PerChannel)"}}), {"PRelu"}};

const auto fusingPReluPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                ngraph::Shape shape(1, 1);
                auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape));
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::LeakyRelu, shape, data);
            }, "PRelu(PerTensor)"}}), {"PRelu"}};

const auto fusingSwish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Swish, {}, {1.0f});
            }, "Swish"}}), {"Swish"}};

const auto fusingSoftPlus = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::SoftPlus, {}, {});
            }, "SoftPlus"}}), {"SoftPlus"}};

const auto fusingHSwish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::HSwish, {}, {});
            }, "HSwish"}}), {"HSwish"}};

const auto fusingMish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Mish, {}, {});
            }, "Mish"}}), {"Mish"}};

const auto fusingHSigmoid = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::HSigmoid);
            }, "HSigmoid"}}), {"HSigmoid"}};

const auto fusingReluAdd = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Relu);
            }, "Relu"},
            {[](postNodeConfig& cfg){
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"}}), {"Relu", "Add"}};

const auto fusingReluScaleShift = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Relu);
            }, "Relu"},
            {[](postNodeConfig& cfg){
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
            }, "Multiply(PerChannel)"},
            {[](postNodeConfig& cfg){
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"}}), {"Relu", "Add"}};

const auto fusingScaleShift = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
            }, "Multiply(PerChannel)"},
            {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingClampRoundAddRelu = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
            }, "Clamp"},
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::RoundHalfToEven);
            }, "RoundHalfToEven"},
            {[](postNodeConfig& cfg){
                ngraph::Shape secondMultInShape(1, 1);
                auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, secondMultInput);
            }, "AddPerTensor"},
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Relu);
            }, "Relu"}}), {"Clamp", "Round", "Add", "Relu"}};

const auto fusingScaleShiftAndFakeQuantizePerChannel = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
            }, "Multiply(PerChannel)"},
            {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"},
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                // auto newShape = ngraph::Shape(cfg.inputNode->get_output_partial_shape(0).size(), 1);
                return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizePerTensor = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ngraph::Shape newShape(cfg.input->get_output_partial_shape(0).size(), 1);
                return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"}}), {"FakeQuantize"} };

const auto fusingFakeQuantizePerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizePerChannelRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ngraph::Shape newShape = generatePerChannelShape(cfg.target);
                return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"},
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Relu);
            }, "Relu"}}), {"FakeQuantize", "Relu"}};

const auto fusingFQPerChannelSigmoidFQPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ngraph::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1].get_length();
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ngraph::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1].get_length();
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize", "Sigmoid", "FakeQuantize"}};

const auto fusingFQPerChannelSigmoidFQPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ngraph::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1].get_length();
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ngraph::Shape newShape(shape.size(), 1);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerTensor)"}}), {"FakeQuantize", "Sigmoid", "FakeQuantize"}};

const auto fusingFakeQuantizePerTensorRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg) {
                auto localPrc = cfg.input->get_element_type();
                auto newShape = ngraph::Shape(cfg.input->get_output_partial_shape(0).size(), 1);
                return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"},
            {[](postNodeConfig& cfg){
                return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Relu);
            }, "Relu"}}), {"FakeQuantize", "Relu"}};

const auto fusingSum = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto shape = cfg.input->get_output_partial_shape(0);
                ov::ParameterVector newParams{std::make_shared<ov::op::v0::Parameter>(cfg.type, shape)};
                cfg.params.insert(cfg.params.end(), newParams.begin(), newParams.end());
                return std::make_shared<ngraph::opset1::Add>(cfg.input, newParams[0]);
            }, "Add(Parameters)"}}), {"Add"}};

const auto fusingSumEluFQ = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto shape = cfg.input->get_output_partial_shape(0);
            ov::ParameterVector newParams{std::make_shared<ov::op::v0::Parameter>(cfg.type, shape)};
            cfg.params.insert(cfg.params.end(), newParams.begin(), newParams.end());
            return std::make_shared<ngraph::opset1::Add>(cfg.input, newParams[0]);
        }, "Add(Parameters)"},
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Elu, {}, {2.0f});
        }, "Elu"},
        {[](postNodeConfig& cfg) {
            auto localPrc = cfg.input->get_element_type();
            auto newShape = ngraph::Shape(cfg.input->get_output_partial_shape(0).size(), 1);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerTensor)"}}), {"Add", "Elu", "FakeQuantize"}};

const auto fusingMultiplyPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::op::v1::Multiply>(cfg.input, secondMultInput);
        }, "Multiply(PerTensor)"}}), {"Multiply"}};

const auto fusingMultiplyPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(cfg.input, secondMultInput);
        }, "Multiply(PerChannel)"}}), {"Multiply"}};

const auto fusingMultiplyAddPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingAddPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, secondMultInput);
        }, "Add(PerTensor)"}}), {"Add"}};

const auto fusingAddPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, secondMultInput);
        }, "Add(PerChannel)"}}), {"Add"}};

const auto fusingSubtractPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerTensor)"}}), {"Subtract"}};

const auto fusingSubtractPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerChannel)"}}), {"Subtract"}};

const auto fusingDividePerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape(1, 1);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerTensor)"}}), {"Divide"}};

const auto fusingDividePerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerChannel)"}}), {"Divide"}};

const auto fusingPRelu1D = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto shape = cfg.input->get_output_partial_shape(0);
            ngraph::Shape newShape({static_cast<size_t>(shape[1].get_length())});
            auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(newShape));
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::LeakyRelu, newShape, data);
        }, "PRelu1D"}}), {"PRelu"}};

const auto fusingPRelu1DScaleShift = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto shape = cfg.input->get_output_partial_shape(0);
            ngraph::Shape newShape({static_cast<size_t>(shape[1].get_length())});
            auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(newShape));
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::LeakyRelu, newShape, data);
        }, "PRelu1D"},
        {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
                ngraph::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
                return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingBias = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
            size_t last_dim = cfg.input->get_output_partial_shape(0).rbegin()->get_length();
            auto bias = ngraph::builder::makeConstant(cfg.type, ngraph::Shape{last_dim}, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, bias);
        }, "fusingBias"}}), {"Add"}};

} // namespace CPUTestUtils
