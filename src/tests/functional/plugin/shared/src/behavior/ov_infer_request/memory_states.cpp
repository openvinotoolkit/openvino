// // Copyright (C) 2018-2023 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

#include "behavior/ov_infer_request/memory_states.hpp"

#include <base/behavior_test_utils.hpp>

#include "blob_factory.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sigmoid.hpp"

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
    std::tie(net, statesToQuery, deviceName, configuration) = GetParam();
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
    ov::Core core = createCoreWithTemplate();
    return core.compile_model(net, deviceName, configuration);
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_QueryState) {
    auto executableNet = prepare_network();
    auto inferReq = executableNet.create_infer_request();

    auto states = inferReq.query_state();
    ASSERT_TRUE(states.size() == 2) << "Incorrect number of VariableStates";

    for (auto&& state : states) {
        auto name = state.get_name();
        ASSERT_TRUE(std::find(statesToQuery.begin(), statesToQuery.end(), name) != statesToQuery.end())
            << "State " << name << "expected to be in memory states but it is not!";
    }
}

TEST_P(OVInferRequestVariableStateTest, inferreq_smoke_VariableState_SetState) {
    auto executableNet = prepare_network();
    auto inferReq = executableNet.create_infer_request();

    const float new_state_val = 13.0f;
    for (auto&& state : inferReq.query_state()) {
        state.reset();
        auto state_val = state.get_state();
        auto element_count = state_val.get_size();
        float* new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        ov::Tensor state_tensor = ov::Tensor(ov::element::f32, ov::Shape({1, element_count}));
        std::memcpy(state_tensor.data(), new_state_data, element_count * sizeof(float));
        delete[] new_state_data;
        state.set_state(state_tensor);
    }

    for (auto&& state : inferReq.query_state()) {
        auto lastState = state.get_state();
        auto last_state_size = lastState.get_size();
        auto last_state_data = static_cast<float*>(lastState.data());
        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";
        for (int i = 0; i < last_state_size; i++) {
            EXPECT_NEAR(new_state_val, last_state_data[i], 1e-5);
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov