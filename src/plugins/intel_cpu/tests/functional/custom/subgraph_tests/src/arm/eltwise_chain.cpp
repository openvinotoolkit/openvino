// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/eltwise_chain.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "custom/single_layer_tests/classes/comparison.hpp"
#include "internal_properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;
using namespace ov::test::eltwise_chain;

namespace {

std::vector<std::vector<EltwiseChainOp>> eltwiseOpsConvertInt8 = {
        { EltwiseTypes::MULTIPLY },
        { EltwiseTypes::ADD },
        { EltwiseTypes::DIVIDE }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_EltwiseChain_MergeConvert_int8,
    EltwiseChainTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesConvert())),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisionsConvert()),
                       ::testing::ValuesIn(eltwiseOpsConvertInt8),
                       ::testing::Values(PostNode::EMPTY),
                       ::testing::ValuesIn({ov::element::i8, ov::element::u8}),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    EltwiseChainTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64)
const InputShape scalar{{}, {{1, 1}}};
const InputShape vector{{}, {{1, 4}}};
const InputShape tail{{}, {{1, 29}}};
const InputShape broadcast{{}, {{2, 29}}};
const InputShape dynamic{{-1, -1}, {{1, 29}, {3, 29}, {5, 1}, {2, 29}}};
const InputShape dynamicBroadcast{{1, -1}, {{1, 29}, {1, 29}, {1, 1}, {1, 29}}};

const std::vector<ElementType> twoI32Inputs(2, ElementType::i32);
const std::vector<ElementType> threeI32Inputs(3, ElementType::i32);
const std::vector<ElementType> fourI32Inputs(4, ElementType::i32);
const std::vector<EltwiseChainOp> mulAdd{EltwiseTypes::MULTIPLY, EltwiseTypes::ADD};
const std::vector<EltwiseChainOp> addMulAdd{EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::ADD};
const std::vector<EltwiseChainOp> addDivide{EltwiseTypes::ADD, EltwiseTypes::DIVIDE};
const std::vector<EltwiseChainOp> comparisonOps{EltwiseTypes::ADD, EltwiseTypes::SUBTRACT, ComparisonTypes::EQUAL};
const std::vector<EltwiseChainOp> multiply{EltwiseTypes::MULTIPLY};
const std::vector<EltwiseChainOp> floor{ActivationTypes::Floor};

EltwiseChainTuple chain_case(std::vector<InputShape> shapes,
                             std::vector<ElementType> precisions,
                             std::vector<EltwiseChainOp> ops,
                             PostNode post = PostNode::EMPTY,
                             ElementType conversion = ElementType::dynamic) {
    return {std::move(shapes),
            InputLayerType::PARAMETER,
            std::move(precisions),
            std::move(ops),
            post,
            conversion,
            DEVICE_CPU};
}

const std::vector<EltwiseChainTuple> chainCases = {
    chain_case({scalar, scalar, scalar}, threeI32Inputs, mulAdd),
    chain_case({vector, vector, vector}, threeI32Inputs, mulAdd),
    chain_case({tail, tail, tail}, threeI32Inputs, mulAdd),
    chain_case({broadcast, scalar, broadcast}, threeI32Inputs, mulAdd),
    chain_case({dynamic, dynamicBroadcast, dynamic}, threeI32Inputs, mulAdd),
    chain_case({tail, tail, tail, tail}, fourI32Inputs, addMulAdd),
    chain_case({tail, tail, tail, tail}, fourI32Inputs, comparisonOps, PostNode::LOGICAL_NOT),
    chain_case({tail, tail}, {ElementType::i8, ElementType::i32}, multiply, PostNode::EMPTY, ElementType::i32),
    chain_case({tail, tail}, {ElementType::u8, ElementType::i32}, multiply, PostNode::EMPTY, ElementType::i32),
    chain_case({tail, tail}, {ElementType::i32, ElementType::i8}, multiply, PostNode::EMPTY, ElementType::i32),
    chain_case({tail, tail}, {ElementType::i32, ElementType::u8}, multiply, PostNode::EMPTY, ElementType::i32),
    chain_case({tail}, {ElementType::f16}, floor, PostNode::EMPTY, ElementType::i32),
    chain_case({tail}, {ElementType::f32}, floor, PostNode::EMPTY, ElementType::i32),
};

const std::vector<EltwiseChainTuple> eltwiseOnlyCases = {
    chain_case({tail, tail, tail}, threeI32Inputs, addDivide),
    chain_case({scalar, scalar}, twoI32Inputs, {EltwiseTypes::ADD}, PostNode::EMPTY, ElementType::f32),
    chain_case({tail, tail}, twoI32Inputs, {EltwiseTypes::ADD}, PostNode::EMPTY, ElementType::f32),
};

using I32ChainParams = std::tuple<EltwiseChainTuple, bool>;

class EltwiseChainI32Test : public EltwiseChainTestBase,
                            public testing::WithParamInterface<I32ChainParams>,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<I32ChainParams>& info) {
        const auto& [chain, snippets] = info.param;
        return get_chain_case_name(chain) + (snippets ? "_Snippets" : "_Eltwise");
    }

protected:
    void SetUp() override {
        const auto& [chain, snippets] = GetParam();
        setup_chain(chain);
        abs_threshold = rel_threshold = 0;
        configuration.insert(ov::intel_cpu::snippets_mode(snippets ? ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK
                                                                   : ov::intel_cpu::SnippetsMode::DISABLE));
        if (std::get<5>(chain) == ElementType::f32) {
            configuration.insert(ov::hint::inference_precision(ov::element::f32));
        }
    }

    void validate() override {
        SubgraphBaseTest::validate();
        const auto& [chain, snippets] = GetParam();
        const auto& [shapes, inputType, precisions, ops, post, conversion, device] = chain;
        const bool divide = ops == addDivide;
        if (conversion == ElementType::f32) {
            selectedType = "jit_f32";
        } else if (post == PostNode::LOGICAL_NOT || is_byte(precisions.front())) {
            selectedType = "jit_(i32|I8)";
        } else if (divide) {
            selectedType = "(jit|ref)_i32";
        } else {
            selectedType = "jit_i32";
        }
        CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});

        if (snippets) {
            CheckNumberOfNodesWithType(compiledModel, "Subgraph", 1);
        } else {
            CheckNumberOfNodesWithType(compiledModel, "Eltwise", divide || post == PostNode::LOGICAL_NOT ? 3 : 1);
        }
        if (conversion == ElementType::i32) {
            CheckNumberOfNodesWithType(compiledModel,
                                       "Convert",
                                       snippets || ov::element::Type(precisions.front()).is_real() ? 1 : 0);
        }
        if (divide) {
            bool hasJit = false;
            bool hasReference = false;
            for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
                const auto& info = node->get_rt_info();
                const auto impl = info.find(ov::exec_model_info::IMPL_TYPE);
                if (impl != info.end()) {
                    const auto& type = impl->second.as<std::string>();
                    hasJit |= type == "jit_i32";
                    hasReference |= type == "ref_i32";
                }
            }
            EXPECT_TRUE(hasJit);
            EXPECT_TRUE(hasReference);
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        inputs.clear();
        const auto& chain = std::get<0>(GetParam());
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto type = parameters[i]->get_element_type();
            inputs.emplace(parameters[i], create_and_fill_tensor(type, shapes[i], input_data(chain, type, i)));
        }
    }

private:
    static bool is_byte(ElementType type) {
        return type == ElementType::i8 || type == ElementType::u8;
    }

    static InputGenerateData input_data(const EltwiseChainTuple& chain, const ov::element::Type& type, size_t port) {
        const auto& [shapes, inputType, precisions, ops, post, conversion, device] = chain;
        const auto seed = static_cast<int32_t>(port + 1);
        if (type == ElementType::i8) {
            return {-128, 256, 1, seed};
        }
        if (type == ElementType::u8) {
            return {0, 256, 1, seed};
        }
        if (type.is_real()) {
            // Negative fractions distinguish truncation before Floor from truncation after Floor.
            return {-127.75, 256, 4, seed};
        }
        if (conversion == ElementType::f32) {
            // Rounding the left input before Add gives a different result from rounding the sum.
            return {port == 0 ? (1 << 24) + 1.0 : 1.0, 0, 1, seed};
        }
        if (conversion == ElementType::i32) {
            return {(1 << 23) + 1, 0, 1, seed};
        }
        if (port == 0 || (ops == comparisonOps && port + 1 == precisions.size())) {
            const auto& window = *::ov::test::comparison::i32Windows().front();
            return {window.start_from, window.range, 1, seed};
        }
        if (ops == addDivide) {
            return {port == 1 ? 1.0 : 2.0, 0, 1, seed};
        }
        // Small operands keep the intermediate results representable.
        return {1, 2, 1, seed};
    }
};

TEST_P(EltwiseChainI32Test, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_I32,
                         EltwiseChainI32Test,
                         ::testing::Combine(::testing::ValuesIn(chainCases), ::testing::Bool()),
                         EltwiseChainI32Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_I32EltwiseOnly,
                         EltwiseChainI32Test,
                         ::testing::Combine(::testing::ValuesIn(eltwiseOnlyCases), ::testing::Values(false)),
                         EltwiseChainI32Test::getTestCaseName);
#endif

}  // namespace
}  // namespace test
}  // namespace ov
