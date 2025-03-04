// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

namespace {
constexpr auto t4_name = "lora/MatMul.B";
constexpr auto t5_name = "lora/MatMul.alpha";
constexpr auto t6_name = "lora/MatMul.A";
constexpr auto netType = ov::element::f32;

enum class StatesPolicy {
    EMPTY_TENSORS,
    RANDOM_TENSORS,
    UNDEFINED
};
std::ostream &operator<<(std::ostream& os, StatesPolicy states_policy) {
    switch (states_policy) {
    case StatesPolicy::EMPTY_TENSORS:
        return os << "empty_tensors";
    case StatesPolicy::RANDOM_TENSORS:
        return os << "random_tensors";
    case StatesPolicy::UNDEFINED:
        return os << "undefined";
    default:
        OPENVINO_THROW("Unexpected states policy");
    }
}

}  // namespace

using LoraPatternParams = std::tuple<ov::element::Type,  // states precision
                                     StatesPolicy>;      // states filling policy

class LoraPatternBaseCPUTest : public SubgraphBaseTest, public testing::WithParamInterface<LoraPatternParams> {
public:
static std::string getTestCaseName(testing::TestParamInfo<LoraPatternParams> obj) {
        ov::element::Type states_precision;
        StatesPolicy states_policy;
        std::tie(states_precision, states_policy) = obj.param;

        std::ostringstream result;
        result << "states_precision=" << states_precision << "_states_policy=" << states_policy;
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::tie(states_precision, states_policy) = this->GetParam();
        init_function();
    }

protected:
    virtual void init_function() = 0;

    std::pair<ov::OutputVector, ov::SinkVector> create_states(const std::vector<ov::PartialShape>& shapes,
                                                              const std::vector<std::string> names) {
        ov::OutputVector state_outs;
        ov::SinkVector assigns;
        auto create_state = [&](const ov::PartialShape& shape, const std::string name) {
            auto variable = std::make_shared<ov::op::util::Variable>(
                ov::op::util::VariableInfo{shape, states_precision, name});
            auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
            auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
            assigns.push_back(assign);
            if (states_precision == netType)
                state_outs.push_back(read_value);
            else
                state_outs.push_back(std::make_shared<ov::op::v0::Convert>(read_value, netType));
        };
        OPENVINO_ASSERT(shapes.size() == names.size());
        for (size_t i = 0; i < shapes.size(); ++i)
            create_state(shapes[i], names[i]);
        return std::make_pair(state_outs, assigns);
    }

    void run_test() {
        switch (states_policy) {
        case StatesPolicy::EMPTY_TENSORS:
            run_test_empty_tensors();
            break;
        case StatesPolicy::RANDOM_TENSORS:
            run_test_random_tensors();
            break;
        default:
            OPENVINO_THROW("Unexpected states policy: ", states_policy);
        }
    }

    void run_test_empty_tensors() {
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

    void run_test_random_tensors() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);

        // use the Template plugin as a reference

        auto compiledReferenceModel = core->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
        auto inferRequestRef = compiledReferenceModel.create_infer_request();
        ASSERT_TRUE(inferRequestRef);

        generate_inputs(targetStaticShapes.front());
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
            inferRequestRef.set_tensor(input.first, input.second);
        }

        constexpr size_t lora_order = 25lu;
        constexpr int infer_count = 6lu;

        std::unordered_map<std::string, ov::Shape> stateShapes;
        std::unordered_map<std::string, ov::Shape> initStateShapes;

        auto&& states = inferRequest.query_state();
        for (auto&& state : states) {
            auto shape = state.get_state().get_shape();
            initStateShapes.insert({state.get_name(), shape});
            std::for_each(shape.begin(), shape.end(), [=](ov::Shape::value_type& x) {
                if (0 == x) {
                    x = lora_order;
                }
            });
            stateShapes.insert({state.get_name(), std::move(shape)});
        }

        for (int i = 0; i < infer_count; ++i) {
            // set states

            if (i == 3) {
                // reset states on the 3rd iteration
                for (auto&& item : states) {
                    item.reset();
                }

                for (auto&& item : inferRequestRef.query_state()) {
                    // Template plugin doesn't support reset state for dynamic shape states
                    item.get_state().set_shape(initStateShapes.at(item.get_name()));
                }
            } else if (!(i & 0x1)) {  // every even call
                // generate and set state tensors
                for (auto&& item : states) {
                    auto&& refStates = inferRequestRef.query_state();
                    using ov::test::utils::InputGenerateData;
                    const auto& shape = stateShapes.at(item.get_name());
                    auto tensor =
                        ov::test::utils::create_and_fill_tensor(states_precision, shape, InputGenerateData{0, 10, 1, i});
                    item.set_state(tensor);
                    auto itr = std::find_if(refStates.begin(), refStates.end(), [&](const ov::VariableState& state) {
                        return state.get_name() == item.get_name();
                    });
                    ASSERT_FALSE(itr == refStates.end());
                    itr->set_state(tensor);
                }
            }

            inferRequest.infer();
            inferRequestRef.infer();
            auto outputs = function->outputs();

            auto tx_result = inferRequest.get_tensor(outputs[0]);
            auto tz_result = inferRequest.get_tensor(outputs[1]);

            auto tx_result_ref = inferRequestRef.get_tensor(outputs[0]);
            auto tz_result_ref = inferRequestRef.get_tensor(outputs[1]);

            ov::test::utils::compare(tx_result_ref, tx_result, 1e-4, 1e-4);
            ov::test::utils::compare(tz_result_ref, tz_result, 1e-4, 1e-4);
        }
    }

    StatesPolicy states_policy = StatesPolicy::UNDEFINED;
    ov::element::Type states_precision = ov::element::dynamic;
};

class LoraPatternMatmulCPUTest : public LoraPatternBaseCPUTest {
protected:
    void init_function() override {
        ov::PartialShape shape_x = {-1, -1, K};
        ov::PartialShape shape_w = {N, K};

        auto param_y = std::make_shared<ov::op::v0::Parameter>(netType, shape_x);
        auto param_w = std::make_shared<ov::op::v0::Parameter>(netType, shape_w);

        // "Main" matrix multiplication from the original transformer model
        auto tx = std::make_shared<ov::op::v0::MatMul>(param_y, param_w, false, true);

        // LoRA parameters from states
        auto states = create_states({{N, -1}, {1, -1}, {-1, K}}, {t4_name, t5_name, t6_name});

        // Apply LoRA parameters to the current activations
        auto t5810 = std::make_shared<ov::op::v0::MatMul>(param_y, states.first[2], false, true);
        auto t5811 = std::make_shared<ov::op::v1::Multiply>(t5810, states.first[1]);
        auto t5812 = std::make_shared<ov::op::v0::MatMul>(t5811, states.first[0], false, true);

        // Mix LoRA part into normally computed activations after the "main" MatMul
        auto tz = std::make_shared<ov::op::v1::Add>(tx, t5812);

        auto result_x = std::make_shared<ov::op::v0::Result>(tx);
        auto result_z = std::make_shared<ov::op::v0::Result>(tz);

        function = std::make_shared<ov::Model>(ov::ResultVector({result_x, result_z}),
                                               states.second,
                                               ov::ParameterVector({param_y, param_w}));
    }

    static constexpr size_t K = 563ul;   // Weights matrix K dimension
    static constexpr size_t N = 2048ul;  // Weights matrix N dimension
};

class LoraPatternConvolutionCPUTest : public LoraPatternBaseCPUTest {
public:
    void init_function() override {
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
        auto states = create_states({{num_channels, -1}, {1, -1}, {-1, num_channels}}, {t4_name, t5_name, t6_name});

        // LoRA pattern with additional Transposes to move channel dimensions into positions where MatMul can be applied
        auto t4940 =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<size_t>{2, 3, 0, 1});

        auto t4941 = std::make_shared<ov::op::v1::Transpose>(param_y, t4940);
        auto t4942 = std::make_shared<ov::op::v0::MatMul>(t4941, states.first[2], false, true);
        auto t4943 = std::make_shared<ov::op::v1::Multiply>(t4942, states.first[1]);
        auto t4944 = std::make_shared<ov::op::v0::MatMul>(t4943, states.first[0], false, true);

        auto t4945 =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<size_t>{2, 3, 0, 1});
        auto t4946 = std::make_shared<ov::op::v1::Transpose>(t4944, t4945);

        // Mix LoRA part into normally computed activations after the "main" MatMul
        auto tz = std::make_shared<ov::op::v1::Add>(tx, t4946);

        auto result_x = std::make_shared<ov::op::v0::Result>(tx);
        auto result_z = std::make_shared<ov::op::v0::Result>(tz);

        function = std::make_shared<ov::Model>(ov::ResultVector({result_x, result_z}),
                                               states.second,
                                               ov::ParameterVector({param_y}));
    }

protected:
    static constexpr size_t num_channels = 64ul;
};

TEST_P(LoraPatternMatmulCPUTest, CompareWithRefs) {
    targetStaticShapes = {{{{1, 20, K}}, {{N, K}}}};
    run_test();
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "LoRA", 1);
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "MatMul", 1);
}

TEST_P(LoraPatternConvolutionCPUTest, CompareWithRefs) {
    targetStaticShapes = {{{1, num_channels, 10, 15}}};
    run_test();
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "LoRA", 1);
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "MatMul", 0);
}

const ov::element::TypeVector states_precisions {ov::element::f32, ov::element::f16};
const std::vector<StatesPolicy> states_policies {StatesPolicy::EMPTY_TENSORS, StatesPolicy::RANDOM_TENSORS};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_LoRA_CPU_MatMul, LoraPatternMatmulCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(states_precisions),
                                 ::testing::ValuesIn(states_policies)),
                         LoraPatternBaseCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_LoRA_CPU_Conv, LoraPatternConvolutionCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(states_precisions),
                                 ::testing::ValuesIn(states_policies)),
                         LoraPatternBaseCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
