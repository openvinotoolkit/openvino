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
                   size_t,                    // pattern type #
                   size_t,                    // LayerNum
                   ExpectedNodes,             // Expected node -> count
                   std::string                // Device name
                   >
    MLPTuple;

static std::shared_ptr<ov::Model> initMLPSubgraph0(std::vector<ov::PartialShape>& inputDynamicShapes,
                                                   std::vector<ElementType>& inputPrecisions,
                                                   size_t numLayers) {
    auto A = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    auto B = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    auto add = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, B);

    std::shared_ptr<Node> current = matmul;

    for (size_t i = 0; i < numLayers; ++i) {
        current = std::make_shared<ov::op::v1::Add>(current, add);
    }

    auto result = std::make_shared<ov::op::v0::Result>(current);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{A, B, add}, "mlp");
}

class MLPTest : public testing::WithParamInterface<MLPTuple>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MLPTuple>& obj) {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        size_t numLayers;
        size_t patternType;
        ExpectedNodes expectedNodes;
        std::string targetName;
        std::tie(inputShapes, inputPrecisions, numLayers, patternType, expectedNodes, targetName) = obj.param;
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
        results << "numLayers=" << numLayers;
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
    ExpectedNodes expectedNodes;
    size_t numLayers;
    size_t patternType;
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::tie(inputShapes, inputPrecisions, numLayers, patternType, expectedNodes, targetDevice) = this->GetParam();

        init_input_shapes(inputShapes);

        if (patternType == 0) {
            function = initMLPSubgraph0(inputDynamicShapes, inputPrecisions, 10);
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

TEST_P(MLPTest, CompareWithRefs) {
    std::vector<InputShape> inputShapes;
    std::vector<ElementType> inputPrecisions;
    std::vector<ElementType> matMulIn0Precisions;
    size_t numLayers;
    size_t patternType;
    ExpectedNodes expectedNodes;
    std::tie(inputShapes, inputPrecisions, numLayers, patternType, expectedNodes, targetDevice) = this->GetParam();

    run();

    for (const auto& node : expectedNodes) {
        CheckNumberOfNodesWithType(compiledModel, node.first, node.second);
    }
}

namespace {

std::vector<std::vector<ov::Shape>> inputShapes = {
    {{2, 8, 16, 64}, {2, 8, 64, 64}, {2, 8, 16, 64}},
    {{1, 384, 64, 64}, {1, 384, 64, 64}, {1, 384, 64, 64}},
    {{2, 64, 16, 80}, {2, 64, 80, 80}, {2, 64, 16, 80}},
    {{3, 96, 16, 64}, {3, 96, 64, 64}, {3, 96, 16, 64}},
    {{2, 192, 16, 160}, {2, 192, 160, 160}, {2, 192, 16, 160}},
    {{2, 4, 16, 8}, {2, 4, 8, 8}, {2, 4, 16, 8}},
    {{1, 204, 13, 212}, {1, 204, 212, 212}, {1, 204, 13, 212}},
};

std::vector<size_t> numLayers = {1, 2, 5, 10};
std::vector<size_t> patternTypes = {0};

INSTANTIATE_TEST_SUITE_P(smoke_MLP,
                         MLPTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                            ::testing::Values(std::vector<ElementType>{ElementType::f32,
                                                                                       ElementType::f32,
                                                                                       ElementType::f32}),
                                            ::testing::ValuesIn(numLayers),
                                            ::testing::ValuesIn(patternTypes),
                                            ::testing::Values(ExpectedNodes{{"Subgraph", 1}}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MLPTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
