// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "internal_properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using ExpectedNodes = std::vector<std::pair<std::string, size_t>>;

typedef std::tuple<std::vector<InputShape>,   // Input shapes
                   std::vector<ElementType>,  // Input precisions
                   std::vector<ElementType>,  // MatMul input #0 precisions
                   size_t,                    // pattern type #
                   ExpectedNodes,             // Expected node -> count
                   std::string                // Device name
                   >
    MHATuple;

static std::shared_ptr<ov::Model> initMHASubgraph0(std::vector<ov::PartialShape>& inputDynamicShapes,
                                                   std::vector<ElementType>& inputPrecisions) {
    ov::ParameterVector paramVect;

    auto transpose0Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    paramVect.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    paramVect.push_back(transpose1Param);

    auto addParam = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    paramVect.push_back(addParam);

    auto transpose2Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    paramVect.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, inputDynamicShapes[0].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<float> mulConstData(ov::shape_size(constantShapes[2]));
    auto mulConst = ov::test::utils::make_constant(inputPrecisions[0], constantShapes[2]);

    std::vector<int64_t> reshape0ConstData = {
        static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0] * inputDynamicShapes[0].get_shape()[1] *
                             inputDynamicShapes[0].get_shape()[2]),
        -1};
    auto reshape0Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[2]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1])};
    auto reshape1Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[4], reshape1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[5], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[6], transpose3ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto mul = std::make_shared<ov::op::v1::Multiply>(transpose1, mulConst);
    const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(transpose0, mul, transA, transB);
    const auto add = std::make_shared<ov::op::v1::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ov::op::v1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, paramVect, "mha");
}

static std::shared_ptr<ov::Model> initMHASubgraph1(std::vector<ov::PartialShape>& inputDynamicShapes,
                                                   std::vector<ElementType>& inputPrecisions) {
    ov::ParameterVector paramVect;

    auto transpose0Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    paramVect.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    paramVect.push_back(transpose1Param);

    auto addParam = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    paramVect.push_back(addParam);

    auto transpose2Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    paramVect.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, inputDynamicShapes[0].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[0], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[1], transpose3ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(transpose0, transpose1, transA, transB);
    const auto add = std::make_shared<ov::op::v1::Add>(matMul0, addParam);
    const auto softMax = std::make_shared<ov::op::v1::Softmax>(add, 3);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(softMax, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, paramVect, "mha");
}

class MHATest : public testing::WithParamInterface<MHATuple>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MHATuple>& obj) {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        size_t patternType;
        ExpectedNodes expectedNodes;
        std::string targetName;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNodes, targetName) = obj.param;
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
        results << "patternType=" << patternType;
        results << "expect=";
        for (const auto& node : expectedNodes) {
            results << node.first << "[" << node.second << "]"
                    << "_";
        }
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_element_type() == ov::element::bf16 || funcInput.get_element_type() == ov::element::f16) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 256;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else {
                tensor = ov::test::utils::create_and_fill_tensor_unique_sequence(funcInput.get_element_type(), targetInputStaticShapes[i], -1, 5);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    size_t patternType;
    ExpectedNodes expectedNodes;
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNodes, targetDevice) =
            this->GetParam();

        init_input_shapes(inputShapes);

        if (patternType == 0) {
            function = initMHASubgraph0(inputDynamicShapes, inputPrecisions);
        } else if (patternType == 1) {
            function = initMHASubgraph1(inputDynamicShapes, inputPrecisions);
        } else {
            FAIL() << "Unsupported MHA pattern type";
        }

        // TODO: try better input data initialization to avoid threshold adjustment
        // TODO: support different precisions on inputs
        if (inputPrecisions[0] == ElementType::bf16) {
            abs_threshold = 0.1f;
            rel_threshold = 10.f;

            configuration.insert({ov::hint::inference_precision(ov::element::bf16)});
        }

        if (inputPrecisions[0] == ElementType::f16)
            configuration.insert({ov::hint::inference_precision(ov::element::f16)});

        // Snippets MHA tokenization has limitations to avoid performance degradations. These limitations depend on
        // target machine. Just for testing, we disable these limitations to allow Snippets to tokenize pattern on all
        // machines for validation.
        if (!configuration.count(ov::intel_cpu::snippets_mode.name())) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
        }
    }
};

TEST_P(MHATest, CompareWithRefs) {
    std::vector<InputShape> inputShapes;
    std::vector<ElementType> inputPrecisions;
    std::vector<ElementType> matMulIn0Precisions;
    size_t patternType;
    ExpectedNodes expectedNodes;
    std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNodes, targetDevice) =
        this->GetParam();

    if (inputPrecisions[0] == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    if (inputPrecisions[0] == ElementType::f16 && !ov::with_cpu_x86_avx512_core_amx_fp16())
        GTEST_SKIP();

    if (!ov::with_cpu_x86_avx512_core())
        GTEST_SKIP();

    run();

    for (const auto& node : expectedNodes) {
        CheckNumberOfNodesWithType(compiledModel, node.first, node.second);
    }
}

namespace {

std::vector<std::vector<ov::Shape>> inputShapes = {
    {{2, 8, 16, 64}, {2, 8, 16, 64}, {2, 1, 1, 8}, {2, 8, 16, 64}},
    {{1, 384, 16, 64}, {1, 384, 16, 64}, {1, 1, 1, 384}, {1, 384, 16, 64}},
    {{2, 64, 16, 80}, {2, 64, 16, 80}, {2, 1, 1, 64}, {2, 64, 16, 80}},
    {{3, 96, 16, 64}, {3, 96, 16, 64}, {3, 1, 1, 96}, {3, 96, 16, 64}},
    {{2, 192, 16, 160}, {2, 192, 16, 160}, {2, 1, 1, 192}, {2, 192, 16, 160}},
    {{2, 4, 16, 8}, {2, 4, 16, 8}, {2, 1, 1, 4}, {2, 4, 16, 8}},
    {{1, 204, 13, 212}, {1, 204, 13, 212}, {1, 1, 1, 204}, {1, 204, 13, 212}},
};

std::vector<std::vector<ElementType>> matMulIn0Precisions = {
    {},
};

std::vector<size_t> patternTypes = {0, 1};

INSTANTIATE_TEST_SUITE_P(smoke_MHA,
                         MHATest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                            ::testing::Values(std::vector<ElementType>{ElementType::f32,
                                                                                       ElementType::f32,
                                                                                       ElementType::f32,
                                                                                       ElementType::f32}),
                                            ::testing::ValuesIn(matMulIn0Precisions),
                                            ::testing::ValuesIn(patternTypes),
                                            ::testing::Values(ExpectedNodes{{"Subgraph", 2}}), // MHA + Decomposed Transpose on input
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MHATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_MHA_BF16,
    MHATest,
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
        ::testing::Values(
            std::vector<ElementType>{ElementType::bf16, ElementType::bf16, ElementType::bf16, ElementType::bf16}),
        ::testing::ValuesIn(matMulIn0Precisions),
        ::testing::ValuesIn(patternTypes),
        ::testing::Values(ExpectedNodes{{"Subgraph", 2}, // MHA + Decomposed Transpose on input
                                        {"Transpose", 1}}),  // Plugin disables tokenization of Transpose on output
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MHATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_MHA_FP16,
    MHATest,
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
        ::testing::Values(
            std::vector<ElementType>{ElementType::f16, ElementType::f16, ElementType::f16, ElementType::f16}),
        ::testing::ValuesIn(matMulIn0Precisions),
        ::testing::ValuesIn(patternTypes),
        ::testing::Values(ExpectedNodes{{"Subgraph", 2}, // MHA + Decomposed Transpose on input
                                        {"Transpose", 1}}),  // Plugin disables tokenization of Transpose on output
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MHATest::getTestCaseName);

}  // namespace

static std::shared_ptr<ov::Model> initMHAQuantSubgraph0(std::vector<ov::PartialShape>& inputDynamicShapes,
                                                        std::vector<ElementType>& inputPrecisions,
                                                        std::vector<ElementType>& matMulIn0Precisions) {
    ov::ParameterVector paramVect;

    auto transpose0Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    paramVect.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    paramVect.push_back(transpose1Param);

    auto addParam = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    paramVect.push_back(addParam);

    auto transpose2Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    paramVect.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<int64_t> reshape0ConstData = {
        static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0] * inputDynamicShapes[0].get_shape()[1] *
                             inputDynamicShapes[0].get_shape()[2]),
        -1};
    auto reshape0Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[2], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[2]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1])};
    auto reshape1Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[3], reshape1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[4], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[5], transpose3ConstData);

    float transA = false;
    float transB = false;

    std::shared_ptr<ov::Node> fakeQuantize0;
    if (matMulIn0Precisions[0] == ElementType::u8)
        fakeQuantize0 = ov::test::utils::make_fake_quantize(transpose0Param,
                                                          inputPrecisions[0],
                                                          256,
                                                          {},
                                                          {0.0f},
                                                          {2.55f},
                                                          {0.0f},
                                                          {2.55f});
    else
        fakeQuantize0 = ov::test::utils::make_fake_quantize(transpose0Param,
                                                          inputPrecisions[0],
                                                          256,
                                                          {},
                                                          {-1.28f},
                                                          {1.27f},
                                                          {-1.28f},
                                                          {1.27f});

    const auto fakeQuantize1 = ov::test::utils::make_fake_quantize(transpose1Param,
                                                                 inputPrecisions[1],
                                                                 256,
                                                                 {},
                                                                 {-1.28f},
                                                                 {1.27f},
                                                                 {-1.28f},
                                                                 {1.27f});
    const auto fakeQuantize2 = ov::test::utils::make_fake_quantize(transpose2Param,
                                                                 inputPrecisions[3],
                                                                 256,
                                                                 {},
                                                                 {-1.28f},
                                                                 {1.27f},
                                                                 {-1.28f},
                                                                 {1.27f});

    std::shared_ptr<ov::Node> fakeQuantize4;

    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize1, transpose1Const);
    const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(transpose0, transpose1, transA, transB);
    const auto fakeQuantize3 =
        ov::test::utils::make_fake_quantize(matMul0, inputPrecisions[0], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
    const auto add = std::make_shared<ov::op::v1::Add>(fakeQuantize3, addParam);
    const auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ov::op::v1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softMax, reshape1Const, true);
    if (matMulIn0Precisions[1] == ElementType::u8)
        fakeQuantize4 = ov::test::utils::make_fake_quantize(reshape1,
                                                          inputPrecisions[0],
                                                          256,
                                                          {},
                                                          {0.0f},
                                                          {0.255f},
                                                          {0.0f},
                                                          {0.255f});
    else
        fakeQuantize4 = ov::test::utils::make_fake_quantize(reshape1,
                                                          inputPrecisions[0],
                                                          256,
                                                          {},
                                                          {-0.128f},
                                                          {0.127f},
                                                          {-0.128f},
                                                          {0.127f});
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize2, transpose2Const);
    const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(fakeQuantize4, transpose2, transA, transB);
    const auto fakeQuantize5 =
        ov::test::utils::make_fake_quantize(matMul1, inputPrecisions[0], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize5, transpose3Const);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, paramVect, "mha");
}

static std::shared_ptr<ov::Model> initMHAQuantSubgraph1(const std::vector<ov::PartialShape>& inputDynamicShapes,
                                                        const std::vector<ElementType>& inputPrecisions,
                                                        const std::vector<ElementType>& matMulIn0Precisions,
                                                        const bool fakeQuantize3Exists) {
    ov::ParameterVector paramVect;

    auto transpose0Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    paramVect.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    paramVect.push_back(transpose1Param);

    auto addParam = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    paramVect.push_back(addParam);

    auto transpose2Param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    paramVect.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[2], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ov::op::v0::Constant::create(ElementType::i64, constantShapes[3], transpose3ConstData);

    std::vector<float> mulConstData(ov::shape_size(constantShapes[4]));
    auto mulConst = ov::test::utils::make_constant(inputPrecisions[0], constantShapes[4]);

    float transA = false;
    float transB = false;

    std::shared_ptr<ov::Node> fakeQuantize0;
    if (matMulIn0Precisions[0] == ElementType::u8)
        fakeQuantize0 = ov::test::utils::make_fake_quantize(transpose0Param,
                                                          inputPrecisions[0],
                                                          256,
                                                          {},
                                                          {0.0f},
                                                          {2.55f},
                                                          {0.0f},
                                                          {2.55f});
    else
        fakeQuantize0 = ov::test::utils::make_fake_quantize(transpose0Param,
                                                          inputPrecisions[0],
                                                          256,
                                                          {},
                                                          {-1.28f},
                                                          {1.27f},
                                                          {-1.28f},
                                                          {1.27f});

    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto fakeQuantize1 = ov::test::utils::make_fake_quantize(transpose1,
                                                                 inputPrecisions[1],
                                                                 256,
                                                                 {},
                                                                 {-1.28f},
                                                                 {1.27f},
                                                                 {-1.28f},
                                                                 {1.27f});
    const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(transpose0, fakeQuantize1, transA, transB);
    const auto mul = std::make_shared<ov::op::v1::Multiply>(addParam, mulConst);
    const auto add = std::make_shared<ov::op::v1::Add>(matMul0, mul);
    const auto softMax = std::make_shared<ov::op::v1::Softmax>(add, 3);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(softMax, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(
        fakeQuantize3Exists
            ? ov::test::utils::make_fake_quantize(matMul1, inputPrecisions[0], 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f})
            : matMul1,
        transpose3Const);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, paramVect, "mha");
}

class MHAQuantTest : public testing::WithParamInterface<MHATuple>,
                     virtual public SubgraphBaseTest,
                     public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MHATuple>& obj) {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        size_t patternType;
        std::string targetName;
        ExpectedNodes expectedNodes;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNodes, targetName) = obj.param;
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
        for (size_t i = 0; i < matMulIn0Precisions.size(); i++) {
            results << "MatMulIn0PRC" << std::to_string(i) << "=" << matMulIn0Precisions[i] << "_";
        }
        results << "patternType=" << patternType;
        results << "expect=";
        for (const auto& node : expectedNodes) {
            results << node.first << "[" << node.second << "]"
                    << "_";
        }
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_element_type().is_real()) {
                tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(funcInput.get_element_type(), targetInputStaticShapes[i], 0.0f, 1.5f);
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 255;
                in_data.resolution = 1;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        abs_threshold = 0.1f;

        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        size_t patternType;
        ExpectedNodes expectedNodes;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNodes, targetDevice) =
            this->GetParam();

        init_input_shapes(inputShapes);

        if (patternType == 0) {
            function = initMHAQuantSubgraph0(inputDynamicShapes, inputPrecisions, matMulIn0Precisions);
        } else if (patternType == 1) {
            function = initMHAQuantSubgraph1(inputDynamicShapes, inputPrecisions, matMulIn0Precisions, true);
        } else if (patternType == 2) {
            function = initMHAQuantSubgraph1(inputDynamicShapes, inputPrecisions, matMulIn0Precisions, false);
        } else {
            FAIL() << "Unsupported MHA pattern type";
        }

        // Snippets MHA tokenization has limitations to avoid performance degradations. These limitations depend on
        // target machine. Just for testing, we disable these limitations to allow Snippets to tokenize pattern on all
        // machines for validation.
        if (!configuration.count(ov::intel_cpu::snippets_mode.name())) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
        }
    }
};

TEST_P(MHAQuantTest, CompareWithRefs) {
    std::vector<InputShape> inputShapes;
    std::vector<ElementType> inputPrecisions;
    std::vector<ElementType> matMulIn0Precisions;
    size_t patternType;
    ExpectedNodes expectedNodes;
    std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNodes, targetDevice) =
        this->GetParam();

    if (inputPrecisions[0] == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    if (!ov::with_cpu_x86_avx512_core_vnni())
        GTEST_SKIP();

    run();

    for (const auto& node : expectedNodes) {
        CheckNumberOfNodesWithType(compiledModel, node.first, node.second);
    }
}

namespace {

std::vector<std::vector<ov::Shape>> inputShapesQuant = {
    {{2, 7, 16, 9}, {2, 7, 16, 9}, {2, 1, 1, 7}, {2, 7, 16, 9}},
    {{2, 8, 16, 64}, {2, 8, 16, 64}, {2, 1, 1, 8}, {2, 8, 16, 64}},
    {{1, 384, 16, 64}, {1, 384, 16, 64}, {1, 1, 1, 384}, {1, 384, 16, 64}},
    {{2, 64, 16, 80}, {2, 64, 16, 80}, {2, 1, 1, 64}, {2, 64, 16, 80}},
    {{3, 96, 16, 64}, {3, 96, 16, 64}, {3, 1, 1, 96}, {3, 96, 16, 64}},
    {{2, 192, 16, 160}, {2, 192, 16, 160}, {2, 1, 1, 192}, {2, 192, 16, 160}},
    {{2, 4, 16, 8}, {2, 4, 16, 8}, {2, 1, 1, 4}, {2, 4, 16, 8}},
    {{1, 204, 13, 212}, {1, 204, 13, 212}, {1, 1, 1, 204}, {1, 204, 13, 212}},
    {{1, 207, 13, 211}, {1, 207, 13, 211}, {1, 1, 1, 207}, {1, 207, 13, 211}},
};

std::vector<std::vector<ElementType>> inputPrecisionsQuant = {
    {ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32},
};

std::vector<std::vector<ElementType>> matMulIn0PrecisionsQuant = {
    {ElementType::i8, ElementType::i8},
    {ElementType::i8, ElementType::u8},
};

INSTANTIATE_TEST_SUITE_P(smoke_MHAQuant_Pattern0,
                         MHAQuantTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesQuant)),
                                            ::testing::ValuesIn(inputPrecisionsQuant),
                                            ::testing::ValuesIn(matMulIn0PrecisionsQuant),
                                            ::testing::Values(0),
                                            ::testing::Values(ExpectedNodes{
                                                {"Subgraph", 5},     // FQs on inputs x 3 + MHA + Deq Mul
                                                {"Transpose", 2}}),  // Decomposed Transpose on input + Transpose between MHA and Deq Mul
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MHAQuantTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MHAQuant_Pattern1,
                         MHAQuantTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesQuant)),
                                            ::testing::ValuesIn(inputPrecisionsQuant),
                                            ::testing::ValuesIn(matMulIn0PrecisionsQuant),
                                            ::testing::Values(1),
                                            ::testing::Values(ExpectedNodes{
                                                {"Subgraph", 4},     // FQ on input x 2 + MHA + Deq Mul
                                                {"Transpose", 2}}),  // Decomposed Transpose on input + Transpose between MHA and Deq Mul
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MHAQuantTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MHAQuant_Pattern2,
                         MHAQuantTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesQuant)),
                                            ::testing::ValuesIn(inputPrecisionsQuant),
                                            ::testing::ValuesIn(matMulIn0PrecisionsQuant),
                                            ::testing::Values(2),
                                            ::testing::Values(ExpectedNodes{{"Subgraph", 3},     // FQ on inputs x 2 + MHA
                                                                            {"Transpose", 1}}),  // Decomposed Transpose on input
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MHAQuantTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
