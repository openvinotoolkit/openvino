// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_chain.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;

std::string EltwiseChainTest::getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple> &obj) {
    std::vector<InputShape> inputShapes;
    InputLayerType secondaryInputType;
    std::vector<ElementType> inputPrecisions;
    std::vector<EltwiseTypes> eltwiseOpTypes;
    bool withQuantization;
    ov::element::Type conversion;
    std::string targetName;
    std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, conversion, targetName) = obj.param;
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
    results << "WithQuant=" << withQuantization << "_";
    if (conversion != ov::element::dynamic) {
        results << "Conversion=" << conversion << "_";
    }
    results << "targetDevice=" << targetName;

    return results.str();
}

ov::Tensor EltwiseChainTest::generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape) {
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
        inputs.insert({funcInput.get_node_shared_ptr(), generate_eltwise_input(
                funcInput.get_element_type(),
                targetInputStaticShapes[i])});
    }
}

void EltwiseChainTest::SetUp() {
    abs_threshold = 0.1f;

    std::vector<InputShape> inputShapes;
    InputLayerType secondaryInputType;
    std::vector<ElementType> inputPrecisions;
    std::vector<EltwiseTypes> eltwiseOpTypes;
    bool withQuantization;
    ov::element::Type conversion;
    std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, conversion, targetDevice) = this->GetParam();

    init_input_shapes(inputShapes);

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
            inputNodes2.push_back(inputNode);
        }
    } else {
        paramVec = ov::ParameterVector {std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes.front())};
        inputNodes1.push_back(conversion == ov::element::dynamic
                                  ? paramVec.front()
                                  : std::dynamic_pointer_cast<ov::Node>(
                                        std::make_shared<ov::op::v0::Convert>(paramVec.front(), conversion)));

        for (size_t i = 1; i < inputPrecisions.size(); i++) {
            std::vector<float> input1Data(ov::shape_size(targetStaticShapes[0][i]));
            inputNodes2.push_back(ov::test::utils::make_constant(
                conversion == ov::element::dynamic ? static_cast<ov::element::Type>(inputPrecisions[i]) : conversion,
                targetStaticShapes[0][i]));
        }
    }

    if (withQuantization) {
        std::vector<std::shared_ptr<ov::Node>> eltwiseOps;
        eltwiseOps.push_back(make_eltwise(inputNodes1[0], inputNodes2[0], eltwiseOpTypes[0]));
        for (size_t i = 1; i < eltwiseOpTypes.size() - 1; i++) {
            eltwiseOps.push_back(make_eltwise(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
        }

        std::vector<size_t> constShape(targetStaticShapes[0][0].size(), 1);
        constShape[1] = targetStaticShapes[0][0][1];
        auto fq = ov::test::utils::make_fake_quantize(eltwiseOps[eltwiseOps.size() - 1],
                                                    ov::element::Type(ov::element::f32),
                                                    256,
                                                    constShape);

        eltwiseOps.push_back(make_eltwise(fq, inputNodes2[eltwiseOpTypes.size() - 1], eltwiseOpTypes[eltwiseOpTypes.size() - 1]));

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};
        function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain_fq");
    } else {
        std::vector<std::shared_ptr<ov::Node>> eltwiseOps;
        eltwiseOps.push_back(make_eltwise(inputNodes1[0], inputNodes2[0], eltwiseOpTypes[0]));
        for (size_t i = 1; i < eltwiseOpTypes.size(); i++) {
            eltwiseOps.push_back(make_eltwise(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
        }

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};
        function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain");
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

std::vector<std::vector<EltwiseTypes>> eltwiseOps() {
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

std::vector<std::vector<EltwiseTypes>> eltwiseOpsConvert() {
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
