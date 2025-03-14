// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

static std::shared_ptr<ov::op::v0::FakeQuantize> createFQ(const std::shared_ptr<ov::Node>& input) {
    auto input_low = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0});
    auto input_high = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{49.4914f});
    auto output_low = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0});
    auto output_high = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{49.4914f});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
}

static std::shared_ptr<ov::Model> makeInteraction(const ElementType inType, const ov::PartialShape& inputShape) {
    bool intraFQ = inType == ElementType::i8;
    auto paramType = intraFQ ? ElementType::f32 : inType;
    std::shared_ptr<ov::op::v0::Parameter> input = std::make_shared<ov::op::v0::Parameter>(paramType, inputShape);
    std::shared_ptr<ov::Node> dense_feature = nullptr;
    if (intraFQ) {
        dense_feature = createFQ(input);
    } else {
        dense_feature = input;
    }
    ov::NodeVector features{dense_feature};
    ov::ParameterVector inputsParams{input};
    const size_t sparse_feature_num = 26;
    for (size_t i = 0; i < sparse_feature_num; i++) {
        auto sparse_input = std::make_shared<ov::op::v0::Parameter>(paramType, inputShape);
        std::shared_ptr<ov::Node> sparse_feat = nullptr;
        if (intraFQ) {
            sparse_feat = createFQ(sparse_input);
        } else {
            sparse_feat = sparse_input;
        }
        features.push_back(sparse_feat);
        inputsParams.push_back(sparse_input);
    }
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(dense_feature);
    auto gather_batch_indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto gather_batch_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto gather_batch = std::make_shared<ov::op::v8::Gather>(shapeof, gather_batch_indices, gather_batch_axis);

    auto gather_feature_indices =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{1});
    auto gather_feature_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
    auto gather_feature = std::make_shared<ov::op::v8::Gather>(shapeof, gather_feature_indices, gather_feature_axis);

    auto reshape_dim2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto reshape_shape = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{gather_batch, reshape_dim2, gather_feature}, 0);

    auto concat1 = std::make_shared<ov::op::v0::Concat>(features, 1);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(concat1, reshape_shape, true);
    std::vector<int32_t> transpose1_value = {0, 2, 1};
    auto transpose1_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, transpose1_value);
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(reshape, transpose1_shape);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, transpose1);
    std::shared_ptr<ov::Node> inter = nullptr;
    if (intraFQ) {
        inter = createFQ(matmul);
    } else {
        inter = matmul;
    }
    std::vector<int32_t> transpose2_value = {1, 2, 0};
    auto transpose2_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, transpose2_value);
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(inter, transpose2_shape);
    std::vector<int32_t> reshape2_value = {729, -1};
    auto reshape2_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, reshape2_value);
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(transpose2, reshape2_shape, true);

    std::vector<int32_t> gather_indices_value;
    for (int i = 1; i < 27; i++) {
        for (int j = 0; j < i; j++) {
            gather_indices_value.push_back(i * 27 + j);
        }
    }
    auto gather_indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{351}, gather_indices_value);
    auto gather_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto gather = std::make_shared<ov::op::v8::Gather>(reshape2, gather_indices, gather_axis);
    auto reshape3_dim1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto reshape3_shape = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{reshape3_dim1, gather_batch}, 0);
    auto reshape3 = std::make_shared<ov::op::v1::Reshape>(gather, reshape3_shape, true);

    std::vector<int32_t> transpose3_value = {1, 0};
    auto transpose3_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, transpose3_value);
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(reshape3, transpose3_shape);

    std::vector<int32_t> reshape4_value = {-1, 351};
    auto reshape4_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, reshape4_value);
    auto reshape4 = std::make_shared<ov::op::v1::Reshape>(transpose3, reshape4_shape, true);
    auto concat2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{dense_feature, reshape4}, 1);
    std::shared_ptr<ov::Model> model;
    if (intraFQ) {
        auto add_const = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{355, 1}, 3);
        auto convert = std::make_shared<ov::op::v0::Convert>(add_const, ov::element::f32);
        auto zp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0);
        auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 1);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto multipy = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        const auto matmul = std::make_shared<ov::op::v0::MatMul>(concat2, multipy);
        model = std::make_shared<ov::Model>(matmul, inputsParams, "interaction");
    } else {
        model = std::make_shared<ov::Model>(concat2, inputsParams, "interaction");
    }
    return model;
}

using InteractionLayerCPUTestParams = std::tuple<ElementType, InputShape>;

class IntertactionCPUTest : public testing::WithParamInterface<InteractionLayerCPUTestParams>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InteractionLayerCPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 15;
            in_data.resolution = 32768;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();
        bool with_bf16 = ov::with_cpu_x86_bfloat16() || with_cpu_x86_avx2_vnni_2();
        if (with_bf16 && (inType == ov::element::bf16 || inType == ov::element::i32)) {
            selectedType = makeSelectedTypeStr("ref_any", ov::element::bf16);
        } else {
            selectedType = makeSelectedTypeStr("ref_any", ov::element::f32);
        }
        targetDevice = ov::test::utils::DEVICE_CPU;
        inputDynamicShapes.push_back(inputShape.first);
        const auto& targetInput = inputShape.second;
        for (size_t i = 0; i < targetInput.size(); i++) {
            targetStaticShapes.push_back(std::vector<ov::Shape>(27, targetInput[i]));
        }

        function = makeInteraction(inType, inputShape.first);
    }
};

TEST_P(IntertactionCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Interaction", 1);
}

using IntertactionCPUTest_FP16 = IntertactionCPUTest;
TEST_P(IntertactionCPUTest_FP16, CompareWithRefs) {
    if (!(ov::with_cpu_x86_avx512_core_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
    rel_threshold = 0.01;
    abs_threshold = 0.0078125;

    run();
    CheckNumberOfNodesWithType(compiledModel, "Interaction", 1);
}

namespace {
const std::vector<ElementType> inPrecisions = {ElementType::f32, ElementType::bf16, ElementType::i32, ElementType::i8};
const std::vector<ElementType> inPrecisions_FP16 = {ElementType::f32, ElementType::i32};
// the model has 27 inputs with same shape
const std::vector<InputShape> input_shapes = {
    // temporarily disable dynamic shape for performance issue
    // // dynamic batch
    // {
    //     {-1, 4},
    //     {{6, 4}, {5, 4}, {3, 4}}
    // },
    // // dynamic shape
    // {
    //     {ov::PartialShape::dynamic(2)},
    //     {{3, 4}, {5, 6}, {7, 8}}
    // },
    // static shape
    {{6, 4}, {{6, 4}}},
    {{3, 4}, {{3, 4}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Interaction,
                         IntertactionCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inPrecisions), ::testing::ValuesIn(input_shapes)),
                         IntertactionCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interaction,
                         IntertactionCPUTest_FP16,
                         ::testing::Combine(::testing::ValuesIn(inPrecisions_FP16), ::testing::ValuesIn(input_shapes)),
                         IntertactionCPUTest::getTestCaseName);
}  // namespace
