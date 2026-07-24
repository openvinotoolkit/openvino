// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using BatchMatMulParams = std::tuple<ov::Shape,  // A shape
                                     ov::Shape,  // B shape
                                     bool,       // transpose A
                                     bool,       // transpose B
                                     ov::element::Type>;

class BatchMatMulTest : public testing::WithParamInterface<BatchMatMulParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchMatMulParams>& obj) {
        const auto& [a_shape, b_shape, tr_a, tr_b, prec] = obj.param;
        std::ostringstream result;
        result << "A=" << ov::test::utils::vec2str(a_shape) << "_";
        result << "B=" << ov::test::utils::vec2str(b_shape) << "_";
        result << "trA=" << tr_a << "_trB=" << tr_b << "_";
        result << "precision=" << prec;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [a_shape, b_shape, tr_a, tr_b, prec] = GetParam();
        auto param_a = std::make_shared<ov::op::v0::Parameter>(prec, a_shape);
        auto param_b = std::make_shared<ov::op::v0::Parameter>(prec, b_shape);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param_a, param_b, tr_a, tr_b);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b}, "BatchMatMul");
        abs_threshold = 100.f;
    }
};

TEST_P(BatchMatMulTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(mlir_BatchMatMul,
                         BatchMatMulTest,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 1536}),
                                            ::testing::Values(ov::Shape{1536, 1536}),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::element::f16)),
                         BatchMatMulTest::getTestCaseName);

// -------- Dynamic-shape MatMul (non-batched, dynamic K) --------

using DynamicMatMulParams = std::tuple<ov::test::InputShape,  // A shape (M, K)
                                       ov::test::InputShape,  // B shape (K, N)
                                       ov::element::Type>;

class DynamicMatMulTest : public testing::WithParamInterface<DynamicMatMulParams>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicMatMulParams>& obj) {
        const auto& [a_shape, b_shape, prec] = obj.param;
        std::ostringstream result;
        result << "A_IS=" << ov::test::utils::partialShape2str({a_shape.first}) << "_";
        result << "A_TS=";
        for (const auto& s : a_shape.second) {
            result << ov::test::utils::vec2str(s) << "_";
        }
        result << "B_IS=" << ov::test::utils::partialShape2str({b_shape.first}) << "_";
        result << "B_TS=";
        for (const auto& s : b_shape.second) {
            result << ov::test::utils::vec2str(s) << "_";
        }
        result << "precision=" << prec;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [a_shape, b_shape, prec] = GetParam();
        init_input_shapes({a_shape, b_shape});
        auto param_a = std::make_shared<ov::op::v0::Parameter>(prec, inputDynamicShapes[0]);
        auto param_b = std::make_shared<ov::op::v0::Parameter>(prec, inputDynamicShapes[1]);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param_a, param_b, false, false);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b}, "DynamicMatMul");
        abs_threshold = 0.05f;
        rel_threshold = 0.05f;
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_shapes) override {
        const auto& model_inputs = function->inputs();
        inputs.clear();
        ov::test::utils::InputGenerateData gen(-0.5, 1, 4);
        for (size_t i = 0; i < model_inputs.size(); ++i) {
            auto tensor = ov::test::utils::create_and_fill_tensor(
                model_inputs[i].get_element_type(), target_input_shapes[i], gen);
            inputs.insert({model_inputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(DynamicMatMulTest, Inference) {
    run();
}

// A: (M=1024, K=?) x B: (K=?, N=1536), K varies between iterations.
INSTANTIATE_TEST_SUITE_P(mlir_DynamicMatMul_dynamicK,
                         DynamicMatMulTest,
                         ::testing::Combine(::testing::Values(ov::test::InputShape{
                                                ov::PartialShape{1024, -1},
                                                {ov::Shape{1024, 768}, ov::Shape{1024, 1536}}}),
                                            ::testing::Values(ov::test::InputShape{
                                                ov::PartialShape{-1, 1536},
                                                {ov::Shape{768, 1536}, ov::Shape{1536, 1536}}}),
                                            ::testing::Values(ov::element::f16)),
                         DynamicMatMulTest::getTestCaseName);

// A: (M=?, K=?) x B: (K=?, N=?), all dims dynamic, all vary between iterations.
INSTANTIATE_TEST_SUITE_P(mlir_DynamicMatMul_allDynamic,
                         DynamicMatMulTest,
                         ::testing::Combine(::testing::Values(ov::test::InputShape{
                                                ov::PartialShape{-1, -1},
                                                {ov::Shape{512, 768}, ov::Shape{1024, 1536}, ov::Shape{128, 256}}}),
                                            ::testing::Values(ov::test::InputShape{
                                                ov::PartialShape{-1, -1},
                                                {ov::Shape{768, 384}, ov::Shape{1536, 1024}, ov::Shape{256, 512}}}),
                                            ::testing::Values(ov::element::f16)),
                         DynamicMatMulTest::getTestCaseName);

}  // namespace
