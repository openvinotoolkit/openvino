// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_chain.hpp"

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/op/convert.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;

std::string EltwiseChainTest::getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple> &obj) {
    const auto& [inputShapes,
                 secondaryInputType,
                 inputPrecisions,
                 eltwiseOpTypes,
                 postNode,
                 conversion,
                 targetName,
                 config] = obj.param;
    std::ostringstream results;

    results << "IS=(";
    for (const auto& shape : inputShapes) {
        results << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    results << ")_TS=(";
    for (const auto& shape : inputShapes) {
        for (const auto& item : shape.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
    }
    for (size_t i = 0; i < inputPrecisions.size(); i++) {
        results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i] << "_";
    }
    for (size_t i = 0; i < eltwiseOpTypes.size(); i++) {
        results << "Op" << std::to_string(i) << "=" << eltwiseOpTypes[i] << "_";
    }
    results << "secondaryInputType=" << secondaryInputType << "_";
    results << "PostNode=" << postNode << "_";
    if (conversion != ov::element::dynamic) {
        results << "Conversion=" << conversion << "_";
    }
    results << "targetDevice=" << targetName;

    if (config) {
        results << "_Precision=" << config->name;
    }
    return results.str();
}

ov::Tensor EltwiseChainTest::generate_eltwise_input(const ov::element::Type& type,
                                                    const ov::Shape& shape,
                                                    size_t port) {
    if (precisionConfig) {
        return create_and_fill_tensor(type, shape, precisionConfig->inputData.at(port));
    }
    struct gen_params {
        uint32_t range;
        int32_t start_from;
        int32_t resolution;

        gen_params(uint32_t range = 10, int32_t start_from = 0, int32_t resolution = 1)
                : range(range), start_from(start_from), resolution(resolution) {}
    };

    gen_params params = type.is_real() ? gen_params(10, 1) : gen_params(10, 10);

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = params.start_from;
    in_data.range = params.range;
    in_data.resolution = params.resolution;
    auto tensor = ov::test::utils::create_and_fill_tensor(type, shape, in_data);
    return tensor;
}

void EltwiseChainTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        inputs.insert({funcInput.get_node_shared_ptr(),
                       generate_eltwise_input(funcInput.get_element_type(), targetInputStaticShapes[i], i)});
    }
}

namespace {
std::shared_ptr<ov::Node> make_chain_op(const ov::Output<ov::Node>& lhs,
                                        const ov::Output<ov::Node>& rhs,
                                        const EltwiseChainOp& op) {
    if (const auto* comparison = std::get_if<ComparisonTypes>(&op)) {
        return make_comparison(lhs, rhs, *comparison);
    }
    if (const auto* activation = std::get_if<ActivationTypes>(&op)) {
        return make_activation(lhs, lhs.get_element_type(), *activation);
    }
    if (const auto* extremum = std::get_if<MinMaxOpType>(&op)) {
        if (*extremum == MinMaxOpType::MINIMUM) {
            return std::make_shared<ov::op::v1::Minimum>(lhs, rhs);
        }
        return std::make_shared<ov::op::v1::Maximum>(lhs, rhs);
    }
    return make_eltwise(lhs, rhs, std::get<EltwiseTypes>(op));
}
}  // namespace

void EltwiseChainTest::SetUp() {
    abs_threshold = 0.1f;
    const auto& [inputShapes,
                 secondaryInputType,
                 inputPrecisions,
                 eltwiseOpTypes,
                 postNode,
                 conversion,
                 _targetDevice,
                 config] = this->GetParam();
    precisionConfig = config;
    if (precisionConfig) {
        abs_threshold = 0;
        rel_threshold = 0;
        configuration.insert(ov::intel_cpu::snippets_mode(precisionConfig->snippetsMode));
        selectedType = precisionConfig->selectedType;
    }
    targetDevice = _targetDevice;
    init_input_shapes(inputShapes);
    for (size_t i = 0; i < eltwiseOpTypes.size(); i++) {
        const bool lastOp = i == eltwiseOpTypes.size() - 1 && postNode != PostNode::SOFTSIGN;
        OPENVINO_ASSERT(lastOp || std::holds_alternative<EltwiseTypes>(eltwiseOpTypes[i]),
                        "A comparison can only be the last operation of the chain");
    }

    ov::ParameterVector paramVec;
    std::vector<std::shared_ptr<ov::Node>> inputNodes1;
    std::vector<std::shared_ptr<ov::Node>> inputNodes2;
    if (secondaryInputType == utils::InputLayerType::PARAMETER) {
        for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
            const auto param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[i], inputDynamicShapes[i]);
            paramVec.push_back(param);

            const auto inputNode =
                (conversion == ov::element::dynamic)
                    ? param
                    : std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(param, conversion));
            if (inputNodes1.empty()) {
                inputNodes1.push_back(inputNode);
            }
            // Precision cases use a distinct operand for each operation.
            if (!precisionConfig || i > 0) {
                inputNodes2.push_back(inputNode);
            }
        }
    } else {
        paramVec = ov::ParameterVector {std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes.front())};
        inputNodes1.push_back(conversion == ov::element::dynamic
                                  ? paramVec.front()
                                  : std::dynamic_pointer_cast<ov::Node>(
                                        std::make_shared<ov::op::v0::Convert>(paramVec.front(), conversion)));

        for (size_t i = 1; i < inputPrecisions.size(); i++) {
            std::vector<float> input1Data(ov::shape_size(targetStaticShapes[0][i]));
            const auto type =
                conversion == ov::element::dynamic ? static_cast<ov::element::Type>(inputPrecisions[i]) : conversion;
            inputNodes2.push_back(precisionConfig ? std::make_shared<ov::op::v0::Constant>(
                                                        generate_eltwise_input(type, targetStaticShapes[0][i], i))
                                                  : ov::test::utils::make_constant(type, targetStaticShapes[0][i]));
        }
    }

    std::vector<std::shared_ptr<ov::Node>> eltwiseOps;
    eltwiseOps.push_back(
        make_chain_op(inputNodes1[0], inputNodes2.empty() ? inputNodes1[0] : inputNodes2[0], eltwiseOpTypes[0]));

    if (postNode == PostNode::QUANTIZE) {
        for (size_t i = 1; i < eltwiseOpTypes.size() - 1; i++) {
            eltwiseOps.push_back(make_chain_op(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
        }

        std::vector<size_t> constShape(targetStaticShapes[0][0].size(), 1);
        constShape[1] = targetStaticShapes[0][0][1];
        auto fq = ov::test::utils::make_fake_quantize(eltwiseOps[eltwiseOps.size() - 1],
                                                    ov::element::Type(ov::element::f32),
                                                    256,
                                                    constShape);

        eltwiseOps.push_back(make_chain_op(fq, inputNodes2[eltwiseOpTypes.size() - 1], eltwiseOpTypes[eltwiseOpTypes.size() - 1]));

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};
        function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain_fq");
    } else if (postNode == PostNode::SOFTSIGN) {
        for (size_t i = 1; i < eltwiseOpTypes.size(); i++) {
            eltwiseOps.push_back(make_chain_op(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
        }

        auto softsign = ov::test::utils::make_activation(eltwiseOps[eltwiseOps.size() - 1], ElementType::i32, ActivationTypes::SoftSign);
        eltwiseOps.push_back(softsign);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};

        function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain_softsign");
    } else if (postNode == PostNode::EMPTY || postNode == PostNode::LOGICAL_NOT) {
        for (size_t i = 1; i < eltwiseOpTypes.size(); i++) {
            eltwiseOps.push_back(make_chain_op(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
        }

        if (postNode == PostNode::LOGICAL_NOT) {
            eltwiseOps.push_back(std::make_shared<ov::op::v1::LogicalNot>(eltwiseOps.back()));
        }
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};
        function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain");
    } else {
        OPENVINO_THROW("Post node of type '", postNode, "' is not supported");
    }
}

void EltwiseChainTest::validate() {
    SubgraphBaseTest::validate();
    if (!precisionConfig) {
        return;
    }
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
    for (const auto& [type, count] : precisionConfig->expectedNodes) {
        CheckNumberOfNodesWithType(compiledModel, type, count);
    }
    std::set<std::string> implementations;
    for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
        const auto& info = node->get_rt_info();
        if (info.count(ov::exec_model_info::IMPL_TYPE)) {
            implementations.insert(info.at(ov::exec_model_info::IMPL_TYPE).as<std::string>());
        }
    }
    for (const auto& implementation : precisionConfig->expectedImplementations) {
        EXPECT_EQ(implementations.count(implementation), 1) << implementation;
    }
}

TEST_P(EltwiseChainTest, CompareWithRefs) {
    run();
}

namespace eltwise_chain {
std::vector<std::vector<ov::Shape>> inputShapes() {
    return {
            {{1, 1,  2,  3},    {1, 1,  2, 3},    {1,  1,  2, 3},    {1, 1,  2, 3}},
            {{1, 48, 5,  6},    {1, 48, 1, 1},    {1,  48, 5, 6},    {1, 1,  5, 6}},
            {{1, 72, 28, 28},   {1, 72, 1, 1},    {1,  72, 1, 1},    {1, 72, 1, 1}},
            {{2, 33, 5,  5},    {2, 33, 5, 5},    {2,  33, 1, 5},    {2, 33, 5, 5}},
            {{1, 2,  3},        {3},              {3},               {3}},
            {{1, 12, 5,  5},    {5, 5},           {12, 5,  5},       {1}},
            {{3, 12, 5,  5},    {1, 12, 5, 1},    {3,  1,  1, 1},    {3, 12, 5, 5}},
            {{1, 1,  1,  1},    {1, 12, 5, 1},    {3,  12, 1, 5},    {3, 12, 5, 1}},
            {{1, 1,  1,  1, 6}, {1, 12, 5, 1, 6}, {3,  12, 1, 5, 1}, {3, 12, 5, 1, 1}}
    };
}

std::vector<std::vector<ElementType>> inputPrecisions() {
    return {
        { ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 },
        { ElementType::i32, ElementType::i32, ElementType::i32, ElementType::i32 }
    };
}

std::vector<std::vector<EltwiseChainOp>> eltwiseOps() {
    return {
            {EltwiseTypes::ADD,    EltwiseTypes::MULTIPLY,     EltwiseTypes::SUBTRACT},
            {EltwiseTypes::DIVIDE, EltwiseTypes::SQUARED_DIFF, EltwiseTypes::ADD}
    };
}

std::vector<std::vector<ov::Shape>> inputShapesConvert() {
    return {
            {{1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}}
    };
}

std::vector<std::vector<EltwiseChainOp>> eltwiseOpsConvert() {
    return {
            {EltwiseTypes::MULTIPLY},
            {EltwiseTypes::ADD},
            {EltwiseTypes::DIVIDE},
            {EltwiseTypes::SUBTRACT},
            {EltwiseTypes::POWER},
    };
}

std::vector<std::vector<ElementType>> inputPrecisionsConvert() {
    return {
            {ElementType::i8,  ElementType::f32, ElementType::f32},
            {ElementType::u8,  ElementType::f32, ElementType::f32},
            {ElementType::i16, ElementType::f32, ElementType::f32},
            {ElementType::u16, ElementType::f32, ElementType::f32},
            {ElementType::i32, ElementType::f32, ElementType::f32},
            {ElementType::f16, ElementType::f32, ElementType::f32},
            {ElementType::f32, ElementType::f32, ElementType::f32},
    };
}
}  // namespace eltwise_chain

}  // namespace test
}  // namespace ov
