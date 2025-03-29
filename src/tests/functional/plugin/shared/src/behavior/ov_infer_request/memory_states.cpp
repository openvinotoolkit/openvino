// // Copyright (C) 2018-2025 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

#include "behavior/ov_infer_request/memory_states.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestVariableStateTest::getTestCaseName(const testing::TestParamInfo<memoryStateParams>& obj) {
    std::ostringstream result;
    std::shared_ptr<ov::Model> net;
    std::string deviceName;
    std::vector<std::string> statesToQuery;
    ov::AnyMap configuration;
    std::tie(net, statesToQuery, deviceName, configuration) = obj.param;
    result << "targetDevice=" << deviceName;
    if (!configuration.empty()) {
        using namespace ov::test::utils;
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
        }
    }
    return result.str();
}

void OVInferRequestVariableStateTest::SetUp() {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::tie(net, statesToQuery, deviceName, configuration) = GetParam();
    OVInferRequestTestBase::SetUp();
}

void OVInferRequestVariableStateTest::TearDown() {
    OVInferRequestTestBase::TearDown();
}

std::shared_ptr<ov::Model> OVInferRequestVariableStateTest::get_network() {
    ov::Shape shape = {1, 200};
    ov::element::Type type = ov::element::f32;

    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
    auto mem_i1 = std::make_shared<ov::op::v0::Constant>(type, shape, 0);
    auto mem_r1 = std::make_shared<ov::op::v3::ReadValue>(mem_i1, "r_1-3");
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(mem_r1, input);

    auto mem_i2 = std::make_shared<ov::op::v0::Constant>(type, shape, 0);
    auto mem_r2 = std::make_shared<ov::op::v3::ReadValue>(mem_i2, "c_1-3");
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(mem_r2, mul1);
    auto mem_w2 = std::make_shared<ov::op::v3::Assign>(mul2, "c_1-3");

    auto mem_w1 = std::make_shared<ov::op::v3::Assign>(mul2, "r_1-3");
    auto sigm = std::make_shared<ov::op::v0::Sigmoid>(mul2);
    sigm->set_friendly_name("sigmod_state");
    sigm->get_output_tensor(0).set_names({"sigmod_state"});
    mem_r1->set_friendly_name("Memory_1");
    mem_r1->get_output_tensor(0).set_names({"Memory_1"});
    mem_w1->add_control_dependency(mem_r1);
    sigm->add_control_dependency(mem_w1);

    mem_r2->set_friendly_name("Memory_2");
    mem_r2->get_output_tensor(0).set_names({"Memory_2"});
    mem_w2->add_control_dependency(mem_r2);
    sigm->add_control_dependency(mem_w2);

    auto function = std::make_shared<ov::Model>(ov::NodeVector{sigm}, ov::ParameterVector{input}, "add_output");
    return function;
}

ov::CompiledModel OVInferRequestVariableStateTest::prepare_network() {
    net->add_output("Memory_1");
    net->add_output("Memory_2");
    ov::Core core = ov::test::utils::create_core();
    return core.compile_model(net, deviceName, configuration);
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_QueryState) {
    auto executable_net = prepare_network();
    auto infer_req = executable_net.create_infer_request();

    auto states = infer_req.query_state();
    ASSERT_TRUE(states.size() == 2) << "Incorrect number of VariableStates";

    for (auto&& state : states) {
        auto name = state.get_name();
        ASSERT_TRUE(std::find(statesToQuery.begin(), statesToQuery.end(), name) != statesToQuery.end())
            << "State " << name << "expected to be in memory states but it is not!";
    }
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_SetState) {
    auto executable_net = prepare_network();
    auto infer_req = executable_net.create_infer_request();

    const float new_state_val = 13.0f;
    for (auto&& state : infer_req.query_state()) {
        state.reset();
        auto state_val = state.get_state();
        auto element_count = state_val.get_size();
        auto state_tensor = ov::Tensor(state_val.get_element_type(), ov::Shape({1, element_count}));
        std::fill_n(state_tensor.data<float>(), element_count, new_state_val);
        state.set_state(state_tensor);
    }

    for (auto&& state : infer_req.query_state()) {
        auto last_state = state.get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());
        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";
        for (int i = 0; i < last_state_size; i++) {
            EXPECT_NEAR(new_state_val, last_state_data[i], 1e-5);
        }
    }
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_Reset) {
    auto executable_net = prepare_network();
    auto infer_req = executable_net.create_infer_request();

    const float new_state_val = 13.0f;
    for (auto&& state : infer_req.query_state()) {
        state.reset();
        auto state_val = state.get_state();
        auto element_count = state_val.get_size();

        auto state_tensor = ov::Tensor(state_val.get_element_type(), ov::Shape({1, element_count}));
        std::fill_n(state_tensor.data<float>(), element_count, new_state_val);
        state.set_state(state_tensor);
    }

    infer_req.query_state().front().reset();

    auto states = infer_req.query_state();
    for (int i = 0; i < states.size(); ++i) {
        auto last_state = states[i].get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";
        if (i == 0) {
            for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(0, last_state_data[j], 1e-5);
            }
        } else {
            for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(new_state_val, last_state_data[j], 1e-5);
            }
        }
    }
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_2infers_set) {
    auto executable_net = prepare_network();
    auto infer_req = executable_net.create_infer_request();
    auto infer_req2 = executable_net.create_infer_request();

    const float new_state_val = 13.0f;
    for (auto&& state : infer_req.query_state()) {
        state.reset();
        auto state_val = state.get_state();
        auto element_count = state_val.get_size();

        auto state_tensor = ov::Tensor(state_val.get_element_type(), ov::Shape({1, element_count}));
        std::fill_n(state_tensor.data<float>(), element_count, new_state_val);
        state.set_state(state_tensor);
    }
    for (auto&& state : infer_req2.query_state()) {
        state.reset();
    }

    auto states = infer_req.query_state();
    auto states2 = infer_req2.query_state();
    for (int i = 0; i < states.size(); ++i) {
        auto last_state = states[i].get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(13.0f, last_state_data[j], 1e-5);
        }
    }
    for (int i = 0; i < states2.size(); ++i) {
        auto last_state = states2[i].get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(0, last_state_data[j], 1e-5);
        }
    }
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_2infers) {
    auto executable_net = prepare_network();
    auto infer_req = executable_net.create_infer_request();
    auto infer_req2 = executable_net.create_infer_request();
    const float new_state_val = 13.0f;

    // set the input data for the network
    auto input = executable_net.input();
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    infer_req.set_tensor(input, tensor);
    // }

    // initial state for 2nd infer request
    for (auto&& state : infer_req2.query_state()) {
        auto state_val = state.get_state();
        auto element_count = state_val.get_size();

        auto state_tensor = ov::Tensor(state_val.get_element_type(), ov::Shape({1, element_count}));
        std::fill_n(state_tensor.data<float>(), element_count, new_state_val);
        state.set_state(state_tensor);
    }

    // reset state for 1st infer request
    for (auto&& state : infer_req.query_state()) {
        state.reset();
    }

    infer_req.infer();
    auto states = infer_req.query_state();
    auto states2 = infer_req2.query_state();
    // check the output and state of 1st request
    auto output_tensor = infer_req.get_tensor("sigmod_state");
    auto output_data = output_tensor.data();
    auto data = static_cast<float*>(output_data);
    for (int i = 0; i < output_tensor.get_size(); i++) {
        EXPECT_NEAR(0.5f, data[i], 1e-5);
    }
    for (int i = 0; i < states.size(); ++i) {
        auto last_state = states[i].get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(0.0, last_state_data[j], 1e-5);
        }
    }

    // // check the output and state of 2nd request
    for (int i = 0; i < states2.size(); ++i) {
        auto last_state = states2[i].get_state();
        auto last_state_size = last_state.get_size();
        auto last_state_data = static_cast<float*>(last_state.data());

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(new_state_val, last_state_data[j], 1e-5);
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov