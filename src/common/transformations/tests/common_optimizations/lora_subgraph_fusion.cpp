// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lora_subgraph_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset15.hpp"
#include "ov_ops/lora_subgraph.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

static constexpr auto netType = ov::element::f32;

std::pair<ov::OutputVector, ov::SinkVector> create_states(const std::vector<ov::PartialShape>& shapes) {
    ov::OutputVector read_values;
    ov::SinkVector assigns;
    size_t idx = 0;
    auto create_state = [&](const ov::PartialShape& shape) {
        auto variable =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{shape, netType, std::to_string(idx++)});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
        auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
        read_values.push_back(read_value);
        assigns.push_back(assign);
    };
    for (const auto& shape : shapes)
        create_state(shape);
    return std::make_pair(read_values, assigns);
}

std::shared_ptr<ov::Node> create_lora_subgraph(const ov::Output<ov::Node>& main_flow,
                                               const ov::Output<ov::Node>& lora_input,
                                               const ov::OutputVector& states,
                                               bool add_transposes,
                                               size_t mul_read_value_idx = 1,
                                               size_t add_data_flow_idx = 0) {
    OPENVINO_ASSERT(states.size() == 3, "get_lora_subgraph expects states size == 3");
    OPENVINO_ASSERT(mul_read_value_idx == 0 || mul_read_value_idx == 1, "mul_read_value_idx must be 0 or 1");
    OPENVINO_ASSERT(add_data_flow_idx == 0 || add_data_flow_idx == 1, "add_data_flow_idx must be 0 or 1");

    auto create_transpose = [](const ov::Output<ov::Node>& input) -> ov::Output<ov::Node> {
        auto constant = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {2, 3, 0, 1});
        return std::make_shared<ov::op::v1::Transpose>(input, constant);
    };

    const auto& mm1_input = add_transposes ? create_transpose(lora_input) : lora_input;
    auto mm1 = std::make_shared<ov::op::v0::MatMul>(mm1_input, states[0], false, true);

    const auto& mul_in_0 = mul_read_value_idx == 0 ? states[1] : mm1->output(0);
    const auto& mul_in_1 = mul_read_value_idx == 0 ? mm1->output(0) : states[1];
    auto mul = std::make_shared<ov::op::v1::Multiply>(mul_in_0, mul_in_1);

    auto mm2 = std::make_shared<ov::op::v0::MatMul>(mul, states[2], false, true);

    const auto& add_sec_input = add_transposes ? create_transpose(mm2) : mm2;
    const auto& add_in_0 = add_data_flow_idx == 0 ? main_flow : add_sec_input;
    const auto& add_in_1 = add_data_flow_idx == 0 ? add_sec_input : main_flow;
    return std::make_shared<ov::op::v1::Add>(add_in_0, add_in_1);
}

class LoraSubgraphFusionTests : public TransformationTestsF {
public:
    LoraSubgraphFusionTests() : TransformationTestsF() {
        // TODO: remove when these flags will be enabled in TransformationTestsF (ticket XXX-98039)
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::NAMES);
    }

    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::LoraSubgraphFusion>();
    }
};

class LoraSubgraphFusionMatMulTests : public LoraSubgraphFusionTests {
public:
    const ov::Dimension K = 563;
    const ov::Dimension N = 2048;
    ov::PartialShape shape_x = {-1, -1, K};
    ov::PartialShape shape_w = {N, K};
    ov::PartialShape shape_state_1 = {-1, K};
    ov::PartialShape shape_state_2 = {1, -1};
    ov::PartialShape shape_state_3 = {N, -1};
};

TEST_F(LoraSubgraphFusionMatMulTests, StandardPattern) {
    {
        auto param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto param_w = std::make_shared<ov::op::v0::Parameter>(netType, shape_w);
        auto main_mm = std::make_shared<ov::op::v0::MatMul>(param_lora, param_w, false, true);
        main_mm->set_friendly_name("main_mm");
        auto states = create_states({shape_state_1, shape_state_2, shape_state_3});
        auto lora_subgraph = create_lora_subgraph(main_mm, param_lora, states.first, false);
        lora_subgraph->set_friendly_name("lora_subgraph");
        model = std::make_shared<Model>(OutputVector{lora_subgraph, main_mm},
                                        states.second,
                                        ParameterVector{param_lora, param_w});
    }
    {
        auto param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto param_w = std::make_shared<ov::op::v0::Parameter>(netType, shape_w);
        auto main_mm = std::make_shared<ov::op::v0::MatMul>(param_lora, param_w, false, true);
        main_mm->set_friendly_name("main_mm");

        auto inner_param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto inner_state_1 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_1);
        auto inner_state_2 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_2);
        auto inner_state_3 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_3);
        auto inner_param_mm = std::make_shared<ov::op::v0::Parameter>(netType, main_mm->get_output_partial_shape(0));

        ov::OutputVector states_outs{inner_state_1, inner_state_2, inner_state_3};
        auto lora_subgraph = create_lora_subgraph(inner_param_mm, inner_param_lora, states_outs, false);
        lora_subgraph->set_friendly_name("lora_subgraph");
        ov::ParameterVector inner_params{inner_param_mm, inner_param_lora, inner_state_1, inner_state_2, inner_state_3};
        auto inner_model = std::make_shared<Model>(OutputVector{lora_subgraph}, inner_params);

        auto states = create_states({shape_state_1, shape_state_2, shape_state_3});
        ov::OutputVector lora_inputs{main_mm, param_lora, states.first[0], states.first[1], states.first[2]};
        auto lora = std::make_shared<ov::op::internal::LoraSubgraph>(lora_inputs, inner_model);
        lora->set_friendly_name("lora_subgraph");

        model_ref =
            std::make_shared<Model>(OutputVector{lora, main_mm}, states.second, ParameterVector{param_lora, param_w});
    }
}

TEST_F(LoraSubgraphFusionMatMulTests, ReshaffledEltwiseInputs) {
    {
        auto param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto param_w = std::make_shared<ov::op::v0::Parameter>(netType, shape_w);
        auto main_mm = std::make_shared<ov::op::v0::MatMul>(param_lora, param_w, false, true);
        main_mm->set_friendly_name("main_mm");

        auto states = create_states({shape_state_1, shape_state_2, shape_state_3});
        auto lora_subgraph = create_lora_subgraph(main_mm, param_lora, states.first, false, 0, 1);
        lora_subgraph->set_friendly_name("lora_subgraph");

        model = std::make_shared<Model>(OutputVector{lora_subgraph, main_mm},
                                        states.second,
                                        ParameterVector{param_lora, param_w});
    }
    {
        auto param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto param_w = std::make_shared<ov::op::v0::Parameter>(netType, shape_w);
        auto main_mm = std::make_shared<ov::op::v0::MatMul>(param_lora, param_w, false, true);
        main_mm->set_friendly_name("main_mm");

        auto inner_param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto inner_state_1 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_1);
        auto inner_state_2 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_2);
        auto inner_state_3 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_3);
        auto inner_param_mm = std::make_shared<ov::op::v0::Parameter>(netType, main_mm->get_output_partial_shape(0));

        ov::OutputVector states_outs{inner_state_1, inner_state_2, inner_state_3};
        auto lora_subgraph = create_lora_subgraph(inner_param_mm, inner_param_lora, states_outs, false, 0, 1);
        lora_subgraph->set_friendly_name("lora_subgraph");
        ov::ParameterVector inner_params{inner_param_mm, inner_param_lora, inner_state_1, inner_state_2, inner_state_3};
        auto inner_model = std::make_shared<Model>(OutputVector{lora_subgraph}, inner_params);

        auto states = create_states({shape_state_1, shape_state_2, shape_state_3});
        ov::OutputVector lora_inputs{main_mm, param_lora, states.first[0], states.first[1], states.first[2]};
        auto lora = std::make_shared<ov::op::internal::LoraSubgraph>(lora_inputs, inner_model);
        lora->set_friendly_name("lora_subgraph");

        model_ref =
            std::make_shared<Model>(OutputVector{lora, main_mm}, states.second, ParameterVector{param_lora, param_w});
    }
}

class LoraSubgraphFusionConvolutionTests : public LoraSubgraphFusionTests {
public:
    const ov::Dimension num_channels = 320;
    ov::PartialShape shape_x = {-1, num_channels, -1, -1};
    ov::PartialShape shape_state_1 = {-1, num_channels};
    ov::PartialShape shape_state_2 = {1, -1};
    ov::PartialShape shape_state_3 = {num_channels, -1};
};

TEST_F(LoraSubgraphFusionConvolutionTests, StandardPattern) {
    {
        auto param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto main_conv = ov::test::utils::make_convolution(param_lora,
                                                           netType,
                                                           {1, 1},
                                                           {1, 1},
                                                           {0, 0},
                                                           {0, 0},
                                                           {1, 1},
                                                           ov::op::PadType::EXPLICIT,
                                                           num_channels.get_length());
        main_conv->set_friendly_name("main_conv");
        auto states = create_states({shape_state_1, shape_state_2, shape_state_3});
        auto lora_subgraph = create_lora_subgraph(main_conv, param_lora, states.first, true);
        lora_subgraph->set_friendly_name("lora_subgraph");
        model =
            std::make_shared<Model>(OutputVector{lora_subgraph, main_conv}, states.second, ParameterVector{param_lora});
    }
    {
        auto param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto main_conv = ov::test::utils::make_convolution(param_lora,
                                                           netType,
                                                           {1, 1},
                                                           {1, 1},
                                                           {0, 0},
                                                           {0, 0},
                                                           {1, 1},
                                                           ov::op::PadType::EXPLICIT,
                                                           num_channels.get_length());
        main_conv->set_friendly_name("main_conv");

        auto inner_param_lora = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto inner_state_1 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_1);
        auto inner_state_2 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_2);
        auto inner_state_3 = std::make_shared<ov::op::v0::Parameter>(netType, shape_state_3);
        auto inner_param_conv =
            std::make_shared<ov::op::v0::Parameter>(netType, main_conv->get_output_partial_shape(0));

        ov::OutputVector states_outs{inner_state_1, inner_state_2, inner_state_3};
        auto lora_subgraph = create_lora_subgraph(inner_param_conv, inner_param_lora, states_outs, true);
        lora_subgraph->set_friendly_name("lora_subgraph");
        ov::ParameterVector inner_params{inner_param_conv,
                                         inner_param_lora,
                                         inner_state_1,
                                         inner_state_2,
                                         inner_state_3};
        auto inner_model = std::make_shared<Model>(OutputVector{lora_subgraph}, inner_params);

        auto states = create_states({shape_state_1, shape_state_2, shape_state_3});
        ov::OutputVector lora_inputs{main_conv, param_lora, states.first[0], states.first[1], states.first[2]};
        auto lora = std::make_shared<ov::op::internal::LoraSubgraph>(lora_inputs, inner_model);
        lora->set_friendly_name("lora_subgraph");

        model_ref = std::make_shared<Model>(OutputVector{lora, main_conv}, states.second, ParameterVector{param_lora});
    }
}
