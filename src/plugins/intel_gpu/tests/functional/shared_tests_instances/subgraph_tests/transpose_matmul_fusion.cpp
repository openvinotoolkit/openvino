// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/transpose_matmul_fusion.hpp"

using namespace ov::test;

namespace ov {
namespace test {

using TransposeMatMulFusionParams = std::tuple<ov::PartialShape,  // input A shapes
                                        ov::PartialShape,         // input B shapes
                                        bool>;                    // is transpose fused?

class TransposeMatMulFusionOnGPU: public testing::WithParamInterface<TransposeMatMulFusionParams>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeMatMulFusionParams> obj) {
        ov::PartialShape input0;
        ov::PartialShape input1;
        bool is_fused;

        std::tie(input0, input1, is_fused) = obj.param;

        std::ostringstream result;
        result << "device=(" << std::string(utils::DEVICE_GPU) << ")_";
        result << ov::test::utils::partialShape2str({input0}) << "_";
        result << ov::test::utils::partialShape2str({input1}) << "_";
        result << "is_fused(" << is_fused << ")";
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ov::PartialShape shape1;
        ov::PartialShape shape2;
        bool is_fused;

        std::tie(shape1, shape2, is_fused) = GetParam();

        InputShape input_shape1 = {shape1, {shape1.get_shape()}};
        InputShape input_shape2 = {shape2, {shape2.get_shape()}};
        init_input_shapes({input_shape1, input_shape2});

        const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape1);
        const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape2);
        const auto order = ov::op::v0::Constant::create(ov::element::i32, Shape{4}, {0, 1, 3, 2});
        const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(param1, order);
        const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(param2, order);
        const auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose1, transpose2, false, false);
        const auto constant = op::v0::Constant::create(element::f32, Shape{1}, {9});
        const auto mul = std::make_shared<ov::op::v1::Multiply>(matmul, constant);
        function = std::make_shared<ov::Model>(mul, ov::ParameterVector{param1, param2});
    }

    void TearDown() override {
        bool is_fused;

        std::tie(std::ignore, std::ignore, is_fused) = GetParam();

        const auto model = compiledModel.get_runtime_model();
        int num_ops = 0;
        for (const auto& node : model->get_ordered_ops()) {
            const auto& rt_info = node->get_rt_info();
            const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
            if (layer_type != "Reorder" && layer_type != "Const") {
                num_ops++;
            }
            if (is_fused) {
                EXPECT_NE(layer_type, "Transpose");
                EXPECT_NE(layer_type, "Permute");
            }
        }
        if (is_fused) {
            ASSERT_EQ(num_ops, 5); // two Inputs, one Eltwise, one MatMul and one Output
        } else {
            ASSERT_EQ(num_ops, 7); // two Inputs, two transpose, one Eltwise, one MatMul and one Output
        }
     }
};

}  // namespace test
}  // namespace ov

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_TransposeMatMulFusion, TransposeMatMulFusionOnGPU,
                         ::testing::Values(
                            TransposeMatMulFusionParams({1, 3, 16, 16}, {1, 3, 16, 16}, true),
                            TransposeMatMulFusionParams({1, 3, 128, 64}, {1, 3, 64, 128}, false)),
                         TransposeMatMulFusionOnGPU::getTestCaseName);

TEST_P(TransposeMatMulFusionOnGPU, CompareWithRefs){
    run();
};

}  // namespace
