// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "cpu_test_utils.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"

using namespace ov::test;

namespace CPUTestUtils {

struct postNodeConfig {
    const std::shared_ptr<ov::Node> target;
    std::shared_ptr<ov::Node> input;
    const ov::element::Type& type;
    ov::ParameterVector& params;
};

struct postNodeBuilder {
    std::function<std::shared_ptr<ov::Node>(postNodeConfig& cfg)> makeNode;
    std::string name;
};

class postOpMgr {
public:
    virtual std::shared_ptr<ov::Node> addPostOps(const ov::element::Type& ngPrc,
                                                 ov::ParameterVector& params,
                                                 const std::shared_ptr<ov::Node>& lastNode) const = 0;
    virtual std::string getFusedOpsNames() const = 0;
    virtual ~postOpMgr() = default;
};

class postFunctionMgr : public postOpMgr {
public:
    postFunctionMgr(std::shared_ptr<ov::Model> function) : _pFunction(std::move(function)) {}
    std::shared_ptr<ov::Node> addPostOps(const ov::element::Type& ngPrc,
                                         ov::ParameterVector& params,
                                         const std::shared_ptr<ov::Node>& lastNode) const override;
    std::string getFusedOpsNames() const override;

private:
    std::shared_ptr<ov::Model> _pFunction;
};

class postNodesMgr : public postOpMgr {
public:
    postNodesMgr(std::vector<postNodeBuilder> postNodes);
    std::shared_ptr<ov::Node> addPostOps(const ov::element::Type& ngPrc,
                                         ov::ParameterVector& params,
                                         const std::shared_ptr<ov::Node>& lastNode) const override;
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
    std::shared_ptr<ov::Node> modifyGraph(const ov::element::Type& ngPrc,
                                          ov::ParameterVector& params,
                                          const std::shared_ptr<ov::Node>& lastNode) override;

    void CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const override;

private:
    void CheckFusingResults(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const;

protected:
    std::shared_ptr<postOpMgr> postOpMgrPtr;
    std::vector<std::string> fusedOps;
    bool checkFusingPosition = true;
    bool expectPostOpsToBeFused = true;
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

static int getFusingAxis(const std::shared_ptr<ov::Node>& node) {
    if (std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node)) {
        return node->get_output_partial_shape(0).size() - 1; // last dimension
    } else if (const auto reduce = std::dynamic_pointer_cast<const ov::op::util::ArithmeticReductionKeepDims>(node)) {
        return getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
    } else if (const auto reduce = std::dynamic_pointer_cast<const ov::op::util::LogicalReductionKeepDims>(node)) {
        return getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
    } else {
        return 1; // second dimension
    }
}

static ov::Shape generatePerChannelShape(const std::shared_ptr<ov::Node>& node) {
    const auto shape = node->get_output_partial_shape(0);
    if (shape.size() == 0)
        OPENVINO_THROW("If shape.size() == 0 then PerTensor fusing tests are N/A");
    if (shape.size() == 1)
        OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
    ov::Shape perChannelShape(shape.size(), 1);
    const auto channelAxis = getFusingAxis(node);
    if (channelAxis >= 0)
        perChannelShape[channelAxis] = shape[channelAxis].get_length();

    return perChannelShape;
}

/* FUSING PATTERNS */
const auto emptyFusingSpec = fusingSpecificParams{nullptr, {}};

const auto fusingRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Relu);
            }, "Relu"}}), {"Relu"}};

const auto fusingElu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Elu, {}, {2.0f});
            }, "Elu"}}), {"Elu"}};

const auto fusingGelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Gelu);
            }, "Gelu"}}), {"Gelu"}};

const auto fusingSigmoid = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Sigmoid);
            }, "Sigmoid"}}), {"Sigmoid"}};

const auto fusingClamp = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Clamp, {}, {3.0f, 6.0f});
            }, "Clamp"}}), {"Clamp"}};

const auto fusingTanh = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Tanh);
            }, "Tanh"}}), {"Tanh"}};

const auto fusingAbs = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Abs);
            }, "Abs"}}), {"Abs"}};

const auto fusingSqrt = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Sqrt);
            }, "Sqrt"}}), {"Sqrt"}};

const auto fusingPReluPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto data = ov::test::utils::generateVector<ov::element::Type_t::f32>(ov::shape_size(newShape));
                return utils::make_activation(cfg.input, cfg.type, utils::LeakyRelu, newShape, data);
            }, "PRelu(PerChannel)"}}), {"PRelu"}};

const auto fusingPReluPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                ov::Shape shape(1, 1);
                auto data = ov::test::utils::generateVector<ov::element::Type_t::f32>(ov::shape_size(shape));
                return utils::make_activation(cfg.input, cfg.type, utils::LeakyRelu, shape, data);
            }, "PRelu(PerTensor)"}}), {"PRelu"}};

const auto fusingSwish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Swish, {}, {1.0f});
            }, "Swish"}}), {"Swish"}};

const auto fusingSoftPlus = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::SoftPlus, {}, {});
            }, "SoftPlus"}}), {"SoftPlus"}};

const auto fusingHSwish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::HSwish, {}, {});
            }, "HSwish"}}), {"HSwish"}};

const auto fusingMish = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Mish, {}, {});
            }, "Mish"}}), {"Mish"}};

const auto fusingHSigmoid = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::HSigmoid);
            }, "HSigmoid"}}), {"HSigmoid"}};

const auto fusingReluAdd = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Relu);
            }, "Relu"},
            {[](postNodeConfig& cfg){
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"}}), {"Relu", "Add"}};

const auto fusingReluScaleShift = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Relu);
            }, "Relu"},
            {[](postNodeConfig& cfg){
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
            }, "Multiply(PerChannel)"},
            {[](postNodeConfig& cfg){
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"}}), {"Relu", "Add"}};

const auto fusingScaleShift = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
            }, "Multiply(PerChannel)"},
            {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingClampRoundAddRelu = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Clamp, {}, {3.0f, 6.0f});
            }, "Clamp"},
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::RoundHalfToEven);
            }, "RoundHalfToEven"},
            {[](postNodeConfig& cfg){
                ov::Shape secondMultInShape(1, 1);
                auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, secondMultInput);
            }, "AddPerTensor"},
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Relu);
            }, "Relu"}}), {"Clamp", "Round", "Add", "Relu"}};

const auto fusingScaleShiftAndFakeQuantizePerChannel = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
            }, "Multiply(PerChannel)"},
            {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
            }, "Add(PerChannel)"},
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                // auto newShape = ov::Shape(cfg.inputNode->get_output_partial_shape(0).size(), 1);
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizePerTensor = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ov::Shape newShape(cfg.input->get_output_partial_shape(0).size(), 1);
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"}}), {"FakeQuantize"} };

const auto fusingFakeQuantizePerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizePerBatch = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                const auto shape = cfg.input->get_output_partial_shape(0);
                ov::Shape perBatchSize(shape.size(), 1);
                perBatchSize[0] = shape[0].get_length();
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, perBatchSize);
            }, "FakeQuantize(PerBatch)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizeFullTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                const auto shape = cfg.input->get_output_partial_shape(0);
                ov::Shape fullTensorShape(shape.size(), 1);
                for (size_t axis = 0; axis < shape.size(); axis++) {
                    fullTensorShape[axis] = shape[axis].get_length();
                }
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, fullTensorShape);
            }, "FakeQuantize(FullTensor)"}}), {"FakeQuantize"}};

const auto fusingFakeQuantizePerChannelRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto localPrc = cfg.input->get_element_type();
                ov::Shape newShape = generatePerChannelShape(cfg.target);
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerChannel)"},
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Relu);
            }, "Relu"}}), {"FakeQuantize", "Relu"}};

const auto fusingFQPerChannelSigmoidFQPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ov::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1].get_length();
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            return utils::make_activation(cfg.input, cfg.type, utils::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ov::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1].get_length();
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize", "Sigmoid", "FakeQuantize"}};

const auto fusingFQPerChannelSigmoidFQPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ov::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1].get_length();
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            return utils::make_activation(cfg.input, cfg.type, utils::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            auto shape = cfg.input->get_output_partial_shape(0);
            if (shape.size() == 1)
                OPENVINO_THROW("If shape.size() == 1 then Granularity can be PerTensor only");
            ov::Shape newShape(shape.size(), 1);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerTensor)"}}), {"FakeQuantize", "Sigmoid", "FakeQuantize"}};

const auto fusingFakeQuantizePerTensorRelu = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg) {
                auto localPrc = cfg.input->get_element_type();
                auto newShape = ov::Shape(cfg.input->get_output_partial_shape(0).size(), 1);
                return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
            }, "FakeQuantize(PerTensor)"},
            {[](postNodeConfig& cfg){
                return utils::make_activation(cfg.input, cfg.type, utils::Relu);
            }, "Relu"}}), {"FakeQuantize", "Relu"}};

const auto fusingSum = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
            {[](postNodeConfig& cfg){
                auto shape = cfg.input->get_output_partial_shape(0);
                ov::ParameterVector newParams{std::make_shared<ov::op::v0::Parameter>(cfg.type, shape)};
                cfg.params.insert(cfg.params.end(), newParams.begin(), newParams.end());
                return std::make_shared<ov::op::v1::Add>(cfg.input, newParams[0]);
            }, "Add(Parameters)"}}), {"Add"}};

const auto fusingSumEluFQ = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto shape = cfg.input->get_output_partial_shape(0);
            ov::ParameterVector newParams{std::make_shared<ov::op::v0::Parameter>(cfg.type, shape)};
            cfg.params.insert(cfg.params.end(), newParams.begin(), newParams.end());
            return std::make_shared<ov::op::v1::Add>(cfg.input, newParams[0]);
        }, "Add(Parameters)"},
        {[](postNodeConfig& cfg){
            return utils::make_activation(cfg.input, cfg.type, utils::Elu, {}, {2.0f});
        }, "Elu"},
        {[](postNodeConfig& cfg) {
            auto localPrc = cfg.input->get_element_type();
            auto newShape = ov::Shape(cfg.input->get_output_partial_shape(0).size(), 1);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerTensor)"}}), {"Add", "Elu", "FakeQuantize"}};

const auto fusingMultiplyPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape(1, 1);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Multiply>(cfg.input, secondMultInput);
        }, "Multiply(PerTensor)"}}), {"Multiply"}};

const auto fusingMultiplyPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Multiply>(cfg.input, secondMultInput);
        }, "Multiply(PerChannel)"}}), {"Multiply"}};

const auto fusingMultiplyAddPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingAddPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape(1, 1);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Add>(cfg.input, secondMultInput);
        }, "Add(PerTensor)"}}), {"Add"}};

const auto fusingAddPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Add>(cfg.input, secondMultInput);
        }, "Add(PerChannel)"}}), {"Add"}};

const auto fusingSubtractPerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape(1, 1);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerTensor)"}}), {"Subtract"}};

const auto fusingSubtractPerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerChannel)"}}), {"Subtract"}};

const auto fusingDividePerTensor = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape(1, 1);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerTensor)"}}), {"Divide"}};

const auto fusingDividePerChannel = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape = generatePerChannelShape(cfg.target);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerChannel)"}}), {"Divide"}};

const auto fusingPRelu1D = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto shape = cfg.input->get_output_partial_shape(0);
            ov::Shape newShape({static_cast<size_t>(shape[1].get_length())});
            auto data = ov::test::utils::generateVector<ov::element::Type_t::f32>(ov::shape_size(newShape));
            return utils::make_activation(cfg.input, cfg.type, utils::LeakyRelu, newShape, data);
        }, "PRelu1D"}}), {"PRelu"}};

const auto fusingPRelu1DScaleShift = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto shape = cfg.input->get_output_partial_shape(0);
            ov::Shape newShape({static_cast<size_t>(shape[1].get_length())});
            auto data = ov::test::utils::generateVector<ov::element::Type_t::f32>(ov::shape_size(newShape));
            return utils::make_activation(cfg.input, cfg.type, utils::LeakyRelu, newShape, data);
        }, "PRelu1D"},
        {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
                ov::Shape newShape = generatePerChannelShape(cfg.input);
                auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
                return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingBias = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
            size_t last_dim = cfg.input->get_output_partial_shape(0).rbegin()->get_length();
            auto bias = ov::test::utils::make_constant(cfg.type, ov::Shape{last_dim});
            return std::make_shared<ov::op::v1::Add>(cfg.input, bias);
        }, "fusingBias"}}), {"Add"}};

} // namespace CPUTestUtils
