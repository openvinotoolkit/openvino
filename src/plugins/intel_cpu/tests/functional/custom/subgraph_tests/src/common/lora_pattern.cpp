// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

class LoraPattern : public SubgraphBaseTest {
public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        auto netType = ov::element::f32;

        ov::PartialShape shape_x = {-1, -1, K};
        ov::PartialShape shape_w = {N, K};

        auto param_y = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto param_w = std::make_shared<ov::op::v0::Parameter>(netType, shape_w);

        // "Main" matrix multiplication from the original transformer model
        auto tx = std::make_shared<ov::op::v0::MatMul>(param_y, param_w, false, true);

        // LoRA parameters from states
        auto variable_t4 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({N, -1}), netType, t4_name});
        auto t4 = std::make_shared<ov::op::v6::ReadValue>(variable_t4);
        auto t4_assign = std::make_shared<ov::op::v6::Assign>(t4, variable_t4);

        auto variable_t5 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({1, -1}), netType, t5_name});
        auto t5 = std::make_shared<ov::op::v6::ReadValue>(variable_t5);
        auto t5_assign = std::make_shared<ov::op::v6::Assign>(t5, variable_t5);

        auto variable_t6 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape({-1, K}), netType, t6_name});
        auto t6 = std::make_shared<ov::op::v6::ReadValue>(variable_t6);
        auto t6_assign = std::make_shared<ov::op::v6::Assign>(t6, variable_t6);

        // Apply LoRA parameters to the current activations
        auto t5810 = std::make_shared<ov::op::v0::MatMul>(param_y, t6, false, true);
        auto t5811 = std::make_shared<ov::op::v1::Multiply>(t5810, t5);
        auto t5812 = std::make_shared<ov::op::v0::MatMul>(t5811, t4, false, true);

        // Mix LoRA part into normally computed activations after the "main" MatMul
        auto tz = std::make_shared<ov::op::v1::Add>(tx, t5812);

        auto result_x = std::make_shared<ov::op::v0::Result>(tx);
        auto result_z = std::make_shared<ov::op::v0::Result>(tz);

        function = std::make_shared<ov::Model>(ov::ResultVector({result_x, result_z}),
                                               ov::SinkVector({t4_assign, t5_assign, t6_assign}),
                                               ov::ParameterVector({param_y, param_w}));
    }

public:
    static constexpr size_t K = 563ul;
    static constexpr size_t N = 2048ul;

    static constexpr auto t4_name = "lora/MatMul.B";
    static constexpr auto t5_name = "lora/MatMul.alpha";
    static constexpr auto t6_name = "lora/MatMul.A";
};

TEST_F(LoraPattern, smoke_empty_states) {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);
    targetStaticShapes = {{{{1, 20, K}}, {{N, K}}}};
    generate_inputs(targetStaticShapes.front());
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }

    inferRequest.infer();
    auto outputs = function->outputs();

    auto tx_result = inferRequest.get_tensor(outputs[0]);
    auto tz_result = inferRequest.get_tensor(outputs[1]);
    ov::test::utils::compare(tx_result, tz_result, 1e-4, 1e-4);
}

} // namespace test
} // namespace ov
