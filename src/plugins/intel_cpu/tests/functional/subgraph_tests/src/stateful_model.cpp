// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
class StaticShapeStatefulModel : public SubgraphBaseTest {
public:
    static constexpr ov::element::Type_t testPrc = ov::element::Type_t::f32;

public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::element::Type netPrc = testPrc;

        const ov::Shape inpShape = {1, 1};
        targetStaticShapes = {{inpShape}};

        auto arg = std::make_shared<ov::op::v0::Parameter>(netPrc, ov::Shape{1, 1});
        auto init_const = ov::op::v0::Constant::create(netPrc, ov::Shape{1, 1}, {0});

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
        std::vector<std::shared_ptr<ov::Node>> args = {arg, read};
        auto add = ngraph::builder::makeEltwise(arg, read, ngraph::helpers::EltwiseTypes::ADD);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto add2 = ngraph::builder::makeEltwise(add, read, ngraph::helpers::EltwiseTypes::ADD);
        auto res = std::make_shared<ov::op::v0::Result>(add2);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg}));
    }

    const std::vector<float>& get_inputs() const {
        static const std::vector<float> input_vals =
            {6.06f, 5.75f, 1.92f, 1.61f, 7.78f, 7.47f, 3.64f, 3.33f, 9.5f, 9.19f};
        return input_vals;
    }

    const std::pair<std::vector<float>, std::vector<float>>& calc_refs() const {
        static const std::pair<std::vector<float>, std::vector<float>> result = {
            {6.06f, 17.87f, 25.54, 29.07f, 38.46f, 53.71f, 64.82, 71.79, 84.62, 103.31f}, // expected_res
            {6.06f, 11.81f, 13.73f, 15.34f, 23.12f, 30.59f, 34.23, 37.56f, 47.06f, 56.25f} // expected_states
        };
        return result;
    }

    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }

    void run_test() {
        auto& input_vals = get_inputs();
        for (size_t i = 0; i < input_vals.size(); ++i) {
            inputs.clear();
            const auto& funcInputs = function->inputs();
            const auto& funcInput = funcInputs.front();
            auto tensor = ov::runtime::Tensor{ov::element::f32, funcInput.get_shape()};
            auto inputData = tensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            inputData[0] = input_vals[i];
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            auto outputTensor = inferRequest.get_output_tensor(0);
            ASSERT_TRUE(outputTensor);
            inferRequest.infer();
            constexpr float rel_diff_threshold = 1e-4f;
            const auto& expected_res = calc_refs().first;
            const float expected_val = expected_res[i];
            const float actual_val = outputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[0];
            ASSERT_TRUE(abs(actual_val - expected_val) / abs(expected_val) < rel_diff_threshold);
            auto states = inferRequest.query_state();
            ASSERT_FALSE(states.empty());
            auto mstate = states.front().get_state();
            ASSERT_TRUE(mstate);
            const auto& expected_states = calc_refs().second;
            const float expected_state_val = expected_states[i];
            const float actual_state_val = mstate.data<ov::element_type_traits<ov::element::f32>::value_type>()[0];
            ASSERT_TRUE(abs(expected_state_val - actual_state_val) / abs(expected_state_val) < rel_diff_threshold);
        }
    }

    void reset_state() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }
};

TEST_F(StaticShapeStatefulModel, smoke_Run_Stateful_Static) {
    prepare();
    run_test();
    reset_state();
    run_test();
}

class DynamicShapeStatefulModel : public SubgraphBaseTest {
public:
    static constexpr ov::element::Type_t testPrc = ov::element::Type_t::f32;

public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::element::Type netPrc = testPrc;

        const ov::Shape inpShape = {1, 1};
        const InputShape input_shape = {{-1, 1}, {{1, 1}, {2, 1}, {4, 1}, {8, 1}, {16, 1}}};
        init_input_shapes({input_shape});

        auto arg = std::make_shared<ov::op::v0::Parameter>(netPrc, inputDynamicShapes.front());
        auto init_param = std::make_shared<ov::op::v0::Parameter>(netPrc, ov::PartialShape{-1, 1});

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(init_param, variable);
        std::vector<std::shared_ptr<ov::Node>> args = {arg, read};
        auto add = ngraph::builder::makeEltwise(arg, read, ngraph::helpers::EltwiseTypes::ADD);
        constexpr int concat_axis = 0;
        auto concat = std::make_shared<ngraph::opset1::Concat>(ov::NodeVector{arg, add}, concat_axis);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto res = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg, init_param}));
    }

    const std::vector<float>& get_inputs() const {
        static const std::vector<float> input_vals =
            {2.44f, 8.06f, 0.59f, 5.21f, 0.29f, 3.33f, 0.36f, 1.75f, 3.52f, 5.46f, 4.55f, 7.13f, 7.35f, 4.81f, 4.24f, 3.60f};
        return input_vals;
    }

    std::vector<float> calc_refs(const ov::Shape& inp_shape, std::vector<float>& vec_state) {
        auto size = inp_shape.front();
        auto& input_vals = get_inputs();
        std::vector<float> input(input_vals.begin(), input_vals.begin() + size);
        std::vector<float> result(input.size(), 0.f);
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = input[i] + vec_state[i];
        }
        result.insert(result.begin(), input.begin(), input.end());
        vec_state = result;
        return result;
    }

    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }

    void run_test() {
        std::vector<float> vec_state = {0};

        auto states = inferRequest.query_state();
        ASSERT_FALSE(states.empty());
        auto init_tensor = ov::runtime::Tensor{testPrc, ov::Shape{1, 1}};
        auto init_data = init_tensor.data<ov::element_type_traits<testPrc>::value_type>();
        init_data[0] = vec_state[0];
        states.front().set_state(init_tensor);

        auto& input_vals = get_inputs();

        for (auto&& shapes : targetStaticShapes) {
            inputs.clear();
            auto &input_shape = shapes.front();
            const auto& funcInputs = function->inputs();
            const auto& funcInput = funcInputs.front();
            auto tensor = ov::runtime::Tensor{testPrc, input_shape};
            auto input_data = tensor.data<ov::element_type_traits<testPrc>::value_type>();
            for (size_t i = 0; i < input_shape.front(); ++i) {
                input_data[i] = input_vals[i];
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            auto outputTensor = inferRequest.get_output_tensor(0);
            ASSERT_TRUE(outputTensor);
            inferRequest.infer();
            auto expected_res = calc_refs(input_shape, vec_state);
            ASSERT_EQ(expected_res.size(), outputTensor.get_shape().front());
            auto actual_res = outputTensor.data<ov::element_type_traits<testPrc>::value_type>();

            constexpr float rel_diff_threshold = 1e-4f;
            for (size_t i = 0; i < expected_res.size(); ++i) {
                const float expected_val = expected_res[i];
                const float actual_val = actual_res[i];
                ASSERT_TRUE(abs(actual_val - expected_val) / abs(expected_val) < rel_diff_threshold);
            }

            auto states = inferRequest.query_state();
            ASSERT_FALSE(states.empty());
            auto mstate = states.front().get_state();
            ASSERT_TRUE(mstate);
            ASSERT_EQ(mstate.get_shape().front(), vec_state.size());
            auto actual_state = mstate.data<ov::element_type_traits<testPrc>::value_type>();

            for (size_t i = 0; i < vec_state.size(); ++i) {
                const float expected_state_val = vec_state[i];
                const float actual_state_val = actual_state[i];
                ASSERT_TRUE(abs(expected_state_val - actual_state_val) / abs(expected_state_val) < rel_diff_threshold);
            }
        }
    }

    void reset_state() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }
};

TEST_F(DynamicShapeStatefulModel, smoke_Run_Stateful_Dynamic) {
    prepare();
    run_test();
    reset_state();
    run_test();
}

} // namespace SubgraphTestsDefinitions