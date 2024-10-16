// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/lora_pattern.hpp"

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {


std::string LoraPatternBase::getTestCaseName(const testing::TestParamInfo<const char*>& obj) {
    auto device_name = obj.param;
    return std::string{"targetDevice="} + device_name; //NOLINT
}

constexpr ov::element::Type LoraPatternBase::netType; //redundant variable definition for C++ prior to C++17

void LoraPatternBase::run_test_empty_tensors() {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);
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

void LoraPatternMatmul::SetUp() {
    targetDevice = this->GetParam();

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

void LoraPatternConvolution::SetUp() {
    targetDevice = this->GetParam();

    ov::PartialShape shape_x = {-1, num_channels, -1, -1};

    auto param_y = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);

    // Original Convolution that is modified by LoRA adapter later
    auto tx = ov::test::utils::make_convolution(param_y,
                                                netType,
                                                {1, 1},
                                                {1, 1},
                                                {0, 0},
                                                {0, 0},
                                                {1, 1},
                                                ov::op::PadType::EXPLICIT,
                                                num_channels);

    // LoRA parameters from states
    auto variable_t4 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape({num_channels, -1}), netType, t4_name});
    auto t4 = std::make_shared<ov::op::v6::ReadValue>(variable_t4);
    auto t4_assign = std::make_shared<ov::op::v6::Assign>(t4, variable_t4);

    auto variable_t5 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape({1, -1}), netType, t5_name});
    auto t5 = std::make_shared<ov::op::v6::ReadValue>(variable_t5);
    auto t5_assign = std::make_shared<ov::op::v6::Assign>(t5, variable_t5);

    auto variable_t6 = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape({-1, num_channels}), netType, t6_name});
    auto t6 = std::make_shared<ov::op::v6::ReadValue>(variable_t6);
    auto t6_assign = std::make_shared<ov::op::v6::Assign>(t6, variable_t6);

    // LoRA pattern with additional Transposes to move channel dimensions into positions where MatMul can be applied
    auto t4940 =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<size_t>{2, 3, 0, 1});

    auto t4941 = std::make_shared<ov::op::v1::Transpose>(param_y, t4940);
    auto t4942 = std::make_shared<ov::op::v0::MatMul>(t4941, t6, false, true);
    auto t4943 = std::make_shared<ov::op::v1::Multiply>(t4942, t5);
    auto t4944 = std::make_shared<ov::op::v0::MatMul>(t4943, t4, false, true);

    auto t4945 =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<size_t>{2, 3, 0, 1});
    auto t4946 = std::make_shared<ov::op::v1::Transpose>(t4944, t4945);

    // Mix LoRA part into normally computed activations after the "main" MatMul
    auto tz = std::make_shared<ov::op::v1::Add>(tx, t4946);

    auto result_x = std::make_shared<ov::op::v0::Result>(tx);
    auto result_z = std::make_shared<ov::op::v0::Result>(tz);

    function = std::make_shared<ov::Model>(ov::ResultVector({result_x, result_z}),
                                           ov::SinkVector({t4_assign, t5_assign, t6_assign}),
                                           ov::ParameterVector({param_y}));
}

}  // namespace test
}  // namespace ov