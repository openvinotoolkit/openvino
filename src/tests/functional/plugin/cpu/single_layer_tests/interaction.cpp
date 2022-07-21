// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include <string>
#include <tuple>

using namespace CPUTestUtils;
using namespace ov::test;
using namespace ngraph;

namespace CPULayerTestsDefinitions {
using InteractionLayerCPUTestParams = std::tuple<ElementType>;

class IntertactionLayerCPUTest : public testing::WithParamInterface<InteractionLayerCPUTestParams>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InteractionLayerCPUTestParams>& obj) {
        ElementType inType;
        std::tie(inType) = obj.param;
        std::ostringstream results;
        results << "Prc=" << inType;
        return results.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 15, 0, 32768);

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        ElementType inType;
        std::tie(inType) = GetParam();
        selectedType = std::string("ref_any_") + InferenceEngine::details::convertPrecision(inType).name();
        targetDevice = CommonTestUtils::DEVICE_CPU;
        targetStaticShapes.push_back({Shape{3, 4}});
        auto dense_feature = std::make_shared<ngraph::opset1::Parameter>(element::f32, PartialShape{3, 4});
        NodeVector features{dense_feature};
        std::vector<float> emb_table_value = {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1.,
            1.5, 0.8, -0.7, -0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7};
        std::vector<int32_t> indices_value = {0, 2, 3, 4};
        std::vector<int32_t> offsets = {0, 2, 2};
        for (size_t i = 0; i < 26; i++) {
            auto emb_table = std::make_shared<opset1::Constant>(element::f32, Shape{5, 4}, emb_table_value);
            auto indices = std::make_shared<opset1::Constant>(element::i32, Shape{4}, indices_value);
            auto offset = std::make_shared<opset1::Constant>(element::i32, Shape{3}, offsets);
            features.push_back(std::make_shared<opset8::EmbeddingBagOffsetsSum>(emb_table, indices, offset));
        }

        auto concat1 = std::make_shared<opset1::Concat>(features, 1);
        std::vector<int32_t> reshape_value = {3, 27, 4};
        auto reshape_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{3}, reshape_value);
        auto reshape = std::make_shared<opset1::Reshape>(concat1, reshape_shape, true);
        std::vector<int32_t> transpose1_value = {0, 2, 1};
        auto transpose1_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{3}, transpose1_value);
        auto transpose1 = std::make_shared<opset1::Transpose>(reshape, transpose1_shape);
        auto matmul = std::make_shared<opset1::MatMul>(reshape, transpose1);
        std::vector<int32_t> transpose2_value = {1, 2, 0};
        auto transpose2_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{3}, transpose2_value);
        auto transpose2 = std::make_shared<opset1::Transpose>(matmul, transpose2_shape);
        std::vector<int32_t> reshape2_value = {729, -1};
        auto reshape2_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{2}, reshape2_value);
        auto reshape2 = std::make_shared<opset1::Reshape>(transpose2, reshape2_shape, true);

        std::vector<int32_t> gather_indices_value;
        for (int i = 1; i < 27; i++) {
            for (int j = 0; j < i; j ++) {
                gather_indices_value.push_back(i * 27 + j);
            }
        }
        auto gather_indices =  std::make_shared<opset1::Constant>(element::i32, Shape{351}, gather_indices_value);
        auto gather_axis =  std::make_shared<opset1::Constant>(element::i32, Shape{}, 0);
        auto gather = std::make_shared<opset8::Gather>(reshape2, gather_indices, gather_axis);
        std::vector<int32_t> reshape3_value = {-1, 3};
        auto reshape3_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{2}, reshape3_value);
        auto reshape3 = std::make_shared<opset1::Reshape>(gather, reshape3_shape, true);

        std::vector<int32_t> transpose3_value = {1, 0};
        auto transpose3_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{2}, transpose3_value);
        auto transpose3 = std::make_shared<opset1::Transpose>(reshape3, transpose3_shape);

        std::vector<int32_t> reshape4_value = {3, 351};
        auto reshape4_shape =  std::make_shared<opset1::Constant>(element::i32, Shape{2}, reshape4_value);
        auto reshape4 = std::make_shared<opset1::Reshape>(transpose3, reshape4_shape, true);
        auto concat2 = std::make_shared<opset1::Concat>(NodeVector{dense_feature, reshape4}, 1);
        auto relu = std::make_shared<opset1::Relu>(concat2);
        function = std::make_shared<ov::Model>(relu, ov::ParameterVector{dense_feature}, "interaction");
    }
};

TEST_P(IntertactionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(compiledModel, "interaction");
}

namespace {
const std::vector<ElementType> inPrecisions = {
        ElementType::f32,
        ElementType::bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_Interaction, IntertactionLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inPrecisions)),
        IntertactionLayerCPUTest::getTestCaseName);
} // namespace


} // namespace CPULayerTestsDefinitions