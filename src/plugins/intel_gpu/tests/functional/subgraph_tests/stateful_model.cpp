// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace ov::test;

namespace SubgraphTestsDefinitions {

static constexpr ov::element::Type_t test_element_type = ov::element::Type_t::f32;
class StatefulModelTest : public SubgraphBaseTest {
public:

    void prepare() {
        configuration = {ov::hint::inference_precision(test_element_type)};
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }

    void reset_state() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }

    void float_compare(const float* expected_res, const float* actual_res, size_t size) {
        constexpr float rel_diff_threshold = 1e-4f;
        for (size_t i = 0; i < size; ++i) {
            const float expected_val = expected_res[i];
            const float actual_val = actual_res[i];
            if (0.f == expected_val) {
                ASSERT_TRUE(abs(actual_val) < rel_diff_threshold);
            } else {
                ASSERT_TRUE(abs(actual_val / expected_val - 1.f) < rel_diff_threshold) << "actual: " << actual_val << " expected: " << expected_val;
            }
        }
    }
};

//     ┌────────┐     ┌───────┐
//     │ Param1 │     │ Const │
//     └───┬────┘     └───┬───┘
//         │              │
//         │         ┌────┴──────┐
//  .......│.........│ ReadValue │
//  .      │         └────┬─────┬┘
//  .      │              │     │
//  .      │   ┌─────┐    │     │
//  .      └───┤ Add ├────┘     │
//  .          └──┬──┘          │
//  .             │             │
//  .             │             │
//  . ┌────────┐  │    ┌─────┐  │
//  ..│ Assign ├──┴────┤ Add ├──┘
//    └────────┘       └──┬──┘
//                        │
//                        │
//                   ┌────┴──────┐
//                   │  Result   │
//                   └───────────┘

class StaticShapeStatefulModel : public StatefulModelTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        ov::element::Type model_et = test_element_type;

        const ov::Shape input_shape = {1, 1};
        targetStaticShapes = {{input_shape}};

        auto arg = std::make_shared<ov::op::v0::Parameter>(model_et, ov::Shape{1, 1});
        auto init_const = ov::op::v0::Constant::create(model_et, ov::Shape{1, 1}, {0});

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
            {6.06f, 17.87f, 25.54f, 29.07f, 38.46f, 53.71f, 64.82f, 71.79f, 84.62f, 103.31f}, // expected_res
            {6.06f, 11.81f, 13.73f, 15.34f, 23.12f, 30.59f, 34.23f, 37.56f, 47.06f, 56.25f} // expected_states
        };
        return result;
    }

    void run_test() {
        auto& input_vals = get_inputs();
        for (size_t i = 0; i < input_vals.size(); ++i) {
            inputs.clear();
            const auto& model_inputs = function->inputs();
            const auto& model_input = model_inputs.front();
            auto tensor = ov::runtime::Tensor{ov::element::f32, model_input.get_shape()};
            auto input_data = tensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            input_data[0] = input_vals[i];
            inputs.insert({model_input.get_node_shared_ptr(), tensor});
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            auto output_tensor = inferRequest.get_output_tensor(0);
            ASSERT_TRUE(output_tensor);
            inferRequest.infer();
            const auto& expected_res = calc_refs().first;
            const float expected_val = expected_res[i];
            const float actual_val = output_tensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[0];
            float_compare(&expected_val, &actual_val, 1);
            auto states = inferRequest.query_state();
            ASSERT_FALSE(states.empty());
            auto mstate = states.front().get_state();
            ASSERT_TRUE(mstate);
            const auto& expected_states = calc_refs().second;
            const float expected_state_val = expected_states[i];
            const float actual_state_val = mstate.data<ov::element_type_traits<ov::element::f32>::value_type>()[0];
            float_compare(&expected_state_val, &actual_state_val, 1);
        }
    }
};

TEST_F(StaticShapeStatefulModel, smoke_Run_Stateful_Static) {
    prepare();
    run_test();
    reset_state();
    run_test();
}

//      ┌────────┐     ┌───────┐
//      │ Param1 │     │ Const ├────────┐
//      └───┬────┘     └───┬───┘        │
//          │              │            │
//          │         ┌────┴──────┐     │
//   .......│.........│ ReadValue │     │
//   .      │         └────┬──────┘     │
//   .      │              │            │
//   .      │   ┌─────┐    │            │
//   .      └───┤ Add ├────┘       ┌────┴──────┐
//   .          └──┬──┘            │ ReadValue │..
//   .             │               └────┬──────┘ .
//   .             │                    │        .
//   . ┌────────┐  │    ┌─────┐         │        .
//   ..│ Assign ├──┴────┤ Add ├─────────┘        .
//     └────────┘       └─────┘                  .
//                       /   \                   .
//                      /     \                  .
//         ┌───────────┐       ┌───────────┐     .
//         │  Result   │       │  Assign   │......
//         └───────────┘       └───────────┘

class StaticShapeTwoStatesModel : public StatefulModelTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        ov::element::Type model_et = test_element_type;

        const ov::Shape input_shape = {1, 1};
        targetStaticShapes = {{input_shape}};

        auto arg = std::make_shared<ov::op::v0::Parameter>(model_et, ov::Shape{1, 1});
        auto init_const = ov::op::v0::Constant::create(model_et, ov::Shape{1, 1}, {5.f});

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        auto variable0 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "variable0"});

        auto variable1 = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "variable1"});

        // Creating ov::Model
        auto read0 = std::make_shared<ov::op::v6::ReadValue>(init_const, variable0);
        auto add = ngraph::builder::makeEltwise(arg, read0, ngraph::helpers::EltwiseTypes::ADD);
        auto assign0 = std::make_shared<ov::op::v6::Assign>(add, variable0);
        auto read1 = std::make_shared<ov::op::v6::ReadValue>(init_const, variable1);
        auto add2 = ngraph::builder::makeEltwise(add, read1, ngraph::helpers::EltwiseTypes::ADD);
        auto assign1 = std::make_shared<ov::op::v6::Assign>(add2, variable1);
        auto res = std::make_shared<ov::op::v0::Result>(add2);
        function = std::make_shared<ov::Model>(
            ov::ResultVector({res}),
            ov::SinkVector({assign0, assign1}),
            ov::ParameterVector({arg}));
    }

    const std::vector<float>& get_inputs() const {
        static const std::vector<float> input_vals =
            {6.06f, 5.75f, 1.92f, 1.61f, 7.78f, 7.47f, 3.64f, 3.33f, 9.5f, 9.19f};
        return input_vals;
    }

    const std::pair<std::vector<float>, std::vector<float>>& calc_refs() const {
        static const std::pair<std::vector<float>, std::vector<float>> result = {
            {11.06f, 16.81f, 18.73f, 20.34f, 28.12f, 35.59f, 39.23f, 42.56f, 52.06f, 61.25f}, // state0
            {16.06f, 32.87f, 51.60f, 71.94f, 100.06f, 135.65f, 174.88f, 217.44f, 269.50f, 330.75f} // state1 and result
        };
        return result;
    }

    void run_test() {
        auto& input_vals = get_inputs();
        for (size_t i = 0; i < input_vals.size(); ++i) {
            inputs.clear();
            const auto& model_inputs = function->inputs();
            const auto& model_input = model_inputs.front();
            auto tensor = ov::runtime::Tensor{test_element_type, model_input.get_shape()};
            auto input_data = tensor.data<ov::element_type_traits<test_element_type>::value_type>();
            input_data[0] = input_vals[i];
            inputs.insert({model_input.get_node_shared_ptr(), tensor});
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            auto output_tensor = inferRequest.get_output_tensor(0);
            ASSERT_TRUE(output_tensor);
            inferRequest.infer();
            std::vector<float> expected_state0;
            std::vector<float> expected_results;
            std::tie(expected_state0, expected_results) = calc_refs();

            auto states = inferRequest.query_state();
            ASSERT_FALSE(states.empty());
            ov::Tensor state0;
            ov::Tensor state1;
            for (auto&& state : states) {
                if ("variable0" == state.get_name()) {
                    state0 = state.get_state();
                }
                if ("variable1" == state.get_name()) {
                    state1 = state.get_state();
                }
            }
            ASSERT_TRUE(state0);
            ASSERT_TRUE(state1);
            auto actual_result = output_tensor.data<ov::element_type_traits<test_element_type>::value_type>();
            float_compare(&expected_state0[i], state0.data<ov::element_type_traits<test_element_type>::value_type>(), 1);
            float_compare(&expected_results[i], state1.data<ov::element_type_traits<test_element_type>::value_type>(), 1);
            float_compare(&expected_results[i], actual_result, 1);
        }
    }
};

TEST_F(StaticShapeTwoStatesModel, smoke_Run_Static_Two_States) {
    prepare();
    run_test();
}

// ┌─────────┐    ┌───────────┐
// │ Param1  │--->│ ReadValue │..
// └───┬──┬──┘    └─────┬─────┘ .
//     │  │             │       .
//     │  │             │       .
//     │  └──┬─────┬────┘       .
//     │     │ Add │            .
//     │     └──┬──┘            .
//     │        │               .
//     │    ┌───┴────┐          .
//     └────┤ Concat │          .
//          └────────┘          .
//             / \              .
//            /   \             .
//   ┌────────┐   ┌────────┐    .
//   │ Result │   │ Assign │.....
//   └────────┘   └────────┘

class DynamicShapeStatefulModel : public StatefulModelTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        ov::element::Type input_et = test_element_type;

        const InputShape input_shape = {{-1, 1}, {{1, 1}, {2, 1}, {4, 1}, {8, 1}, {16, 1}}};
        init_input_shapes({input_shape});

        auto arg = std::make_shared<ov::op::v0::Parameter>(input_et, inputDynamicShapes.front());

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(arg, variable);
        std::vector<std::shared_ptr<ov::Node>> args = {arg, read};
        auto add = ngraph::builder::makeEltwise(arg, read, ngraph::helpers::EltwiseTypes::ADD);
        constexpr int concat_axis = 0;
        auto concat = std::make_shared<ngraph::opset1::Concat>(ov::NodeVector{arg, add}, concat_axis);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto res = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg}));
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

    void run_test() {
        std::vector<float> vec_state = {0};

        auto states = inferRequest.query_state();
        ASSERT_FALSE(states.empty());
        auto init_tensor = ov::runtime::Tensor{test_element_type, ov::Shape{1, 1}};
        auto init_data = init_tensor.data<ov::element_type_traits<test_element_type>::value_type>();
        init_data[0] = vec_state[0];
        states.front().set_state(init_tensor);

        auto& input_vals = get_inputs();

        for (auto&& shapes : targetStaticShapes) {
            inputs.clear();
            auto &input_shape = shapes.front();
            const auto& model_inputs = function->inputs();
            const auto& model_input = model_inputs.front();
            auto tensor = ov::runtime::Tensor{test_element_type, input_shape};
            auto input_data = tensor.data<ov::element_type_traits<test_element_type>::value_type>();
            for (size_t i = 0; i < input_shape.front(); ++i) {
                input_data[i] = input_vals[i];
            }
            inputs.insert({model_input.get_node_shared_ptr(), tensor});
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            auto output_tensor = inferRequest.get_output_tensor(0);
            ASSERT_TRUE(output_tensor);
            inferRequest.infer();
            auto expected_res = calc_refs(input_shape, vec_state);
            ASSERT_EQ(expected_res.size(), output_tensor.get_shape().front());
            auto actual_res = output_tensor.data<ov::element_type_traits<test_element_type>::value_type>();

            float_compare(expected_res.data(), actual_res, expected_res.size());

            auto states = inferRequest.query_state();
            ASSERT_FALSE(states.empty());
            auto mstate = states.front().get_state();
            ASSERT_TRUE(mstate);
            ASSERT_EQ(mstate.get_shape().front(), vec_state.size());
            auto actual_state = mstate.data<ov::element_type_traits<test_element_type>::value_type>();

            float_compare(vec_state.data(), actual_state, vec_state.size());
        }
    }
};

TEST_F(DynamicShapeStatefulModel, smoke_Run_Stateful_Dynamic) {
    prepare();
    run_test();
    reset_state();
    run_test();
}

//     ┌────────┐    ┌────────┐
//     │ Param1 │    │ Param2 │
//     └────────┘    └────────┘
//             \     /
//            ┌───────┐
//            │  Add1 │
//            └───┬───┘
//            /   │
//    ┌─────────┐ │  ┌───────┐
//    │ Result1 │ │  │ Const │
//    └───┬─────┘ │  └───────┘
//        │       │  /
//        │    ┌──┴───┐
//        │    │ Add2 │
//        │    └──┬───┘
//        │       │
//        │       │  ┌───────────┐
//        └───────┼─►│ ReadValue │..
//                │  └───────────┘ .
//                │   /            .
//            ┌───┴────┐            .
//            │ Concat │            .
//            └────────┘            .
//             /     \             .
//    ┌─────────┐    ┌────────┐    .
//    │ Result2 │    │ Assign │.....
//    └─────────┘    └────────┘

class DynamicShapeStatefulModelStateAsInp : public StatefulModelTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        ov::element::Type model_et = test_element_type;

        const InputShape input_shape = {{1, -1}, {{1, 1}, {1, 2}, {1, 4}, {1, 8}, {1, 16}}};
        init_input_shapes({input_shape, input_shape});

        auto param1 = std::make_shared<ov::op::v0::Parameter>(model_et, inputDynamicShapes[0]);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(model_et, inputDynamicShapes[1]);
        auto init_param = std::make_shared<ov::op::v0::Parameter>(model_et, ov::PartialShape{1, -1});

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(init_param, variable);
        std::vector<std::shared_ptr<ov::Node>> args = {param1, param2, read};
        auto add1 = ngraph::builder::makeEltwise(param1, param2, ngraph::helpers::EltwiseTypes::ADD);
        auto add_const = ov::op::v0::Constant::create(model_et, ov::Shape{1, 1}, {const_val});
        auto add2 = ngraph::builder::makeEltwise(add1, add_const, ngraph::helpers::EltwiseTypes::ADD);
        constexpr int concat_axis = 1;
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{add2, read}, concat_axis);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto res1 = std::make_shared<ov::op::v0::Result>(add1);
        auto res2 = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(
            ov::ResultVector({res1, res2}),
            ov::SinkVector({assign}),
            ov::ParameterVector({param1, param2, init_param}));
    }

    const std::vector<float>& get_inputs() const {
        static const std::vector<float> input_vals =
            {2.44f, 8.06f, 0.59f, 5.21f, 0.29f, 3.33f, 0.36f, 1.75f, 3.52f, 5.46f, 4.55f, 7.13f, 7.35f, 4.81f, 4.24f, 3.60f};
        return input_vals;
    }

    std::tuple<std::vector<float>, std::vector<float>>
    calc_refs(const ov::Shape& inp_shape, std::vector<float>& vec_state) {
        auto size = inp_shape[1];
        auto& input_vals = get_inputs();
        std::vector<float> input1(input_vals.begin(), input_vals.begin() + size);
        std::vector<float> input2(input_vals.begin(), input_vals.begin() + size);
        std::vector<float> result1(input1.size(), 0.f);
        for (size_t i = 0; i < input1.size(); ++i) {
            result1[i] = input1[i] + input2[i];
        }

        std::vector<float> result2 = result1;
        for (size_t i = 0; i < result2.size(); ++i) {
            result2[i] += const_val;
        }
        result2.insert(result2.end(), vec_state.begin(), vec_state.end());
        vec_state = result2;
        return {result1, result2};
    }

    void run_test() {
        std::vector<float> vec_state = {0.f};

        auto states = inferRequest.query_state();
        ASSERT_FALSE(states.empty());
        auto init_tensor = ov::runtime::Tensor{test_element_type, ov::Shape{1, 1}};
        auto init_data = init_tensor.data<ov::element_type_traits<test_element_type>::value_type>();
        init_data[0] = vec_state[0];
        states.front().set_state(init_tensor);

        auto& input_vals = get_inputs();

        for (auto&& shapes : targetStaticShapes) {
            inputs.clear();
            auto &input_shape = shapes.front();
            const auto& model_inputs = function->inputs();
            for (auto&& model_input : model_inputs) {
                auto tensor = ov::runtime::Tensor{test_element_type, input_shape};
                auto input_data = tensor.data<ov::element_type_traits<test_element_type>::value_type>();
                for (size_t i = 0; i < input_shape[1]; ++i) {
                    input_data[i] = input_vals[i];
                }
                inputs.insert({model_input.get_node_shared_ptr(), tensor});
            }

            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            auto output_tensor1 = inferRequest.get_output_tensor(0);
            auto output_tensor2 = inferRequest.get_output_tensor(1);
            ASSERT_TRUE(output_tensor1);
            ASSERT_TRUE(output_tensor2);
            inferRequest.infer();
            std::vector<float> result1;
            std::vector<float> result2;
            std::tie(result1, result2) = calc_refs(input_shape, vec_state);
            ASSERT_EQ(result1.size(), output_tensor1.get_shape()[1]);
            ASSERT_EQ(result2.size(), output_tensor2.get_shape()[1]);
            auto actual_res1 = output_tensor1.data<ov::element_type_traits<test_element_type>::value_type>();
            auto actual_res2 = output_tensor2.data<ov::element_type_traits<test_element_type>::value_type>();

            float_compare(result1.data(), actual_res1, result1.size());
            float_compare(result2.data(), actual_res2, result2.size());

            auto states = inferRequest.query_state();
            ASSERT_FALSE(states.empty());
            auto mstate = states.front().get_state();
            ASSERT_TRUE(mstate);
            ASSERT_EQ(mstate.get_shape()[1], vec_state.size());
            auto actual_state = mstate.data<ov::element_type_traits<test_element_type>::value_type>();

            float_compare(vec_state.data(), actual_state, vec_state.size());

            states.front().set_state(output_tensor1);
            vec_state = result1;
        }
    }

private:
    const float const_val = 42.0f;
};

TEST_F(DynamicShapeStatefulModelStateAsInp, smoke_Run_Stateful_Dynamic_State_As_Inp) {
    prepare();
    run_test();
    reset_state();
    run_test();
}

} // namespace SubgraphTestsDefinitions
