// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/auto_unit_test.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_ivariable_state.hpp"
using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<bool, ov::AnyMap>;

class AutoLifeTimeTest : public tests::AutoTest, public ::testing::Test {
public:
    void SetUp() override {
        plugin->set_device_name("AUTO");
        mock_compiled_model = {mockIExeNetActual, std::make_shared<std::string>("for test")};
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>("GPU.0"),
                              _))
            .WillByDefault(Return(mock_compiled_model));
        mock_states = {ov::SoPtr<ov::IVariableState>(std::make_shared<NiceMock<ov::MockIVariableState>>(),
                                                     std::make_shared<std::string>("for test"))};
        EXPECT_CALL(*inferReqInternalActual, query_state()).WillRepeatedly(Return(mock_states));
    }

    void TearDown() override {
        testing::Mock::AllowLeak(mock_states.front()._ptr.get());
        testing::Mock::AllowLeak(inferReqInternalActual.get());
    }

protected:
    ov::SoPtr<ov::MockICompiledModel> mock_compiled_model;
    std::vector<ov::SoPtr<ov::IVariableState>> mock_states;
};

TEST_F(AutoLifeTimeTest, loaded_tensor) {
    // get Parameter
    config.insert(ov::device::priorities("GPU.0"));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    auto request = compiled_model->create_infer_request();
    for (auto& iter : request->get_inputs()) {
        auto tensor = request->get_tensor(iter);
        ASSERT_EQ(tensor._so, mock_compiled_model._so);
    }
}

TEST_F(AutoLifeTimeTest, loaded_states) {
    // get Parameter
    config.insert(ov::device::priorities("GPU.0"));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    auto request = compiled_model->create_infer_request();
    auto states = request->query_state();
    auto res_so = mock_states.front()._so;
    for (auto& state : states)
        ASSERT_EQ(state._so, res_so);
}

TEST_F(AutoLifeTimeTest, loaded_tensor_multi) {
    plugin->set_device_name("MULTI");
    // get Parameter
    config.insert(ov::device::priorities("GPU.0"));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    auto request = compiled_model->create_infer_request();
    for (auto& iter : request->get_inputs()) {
        auto tensor = request->get_tensor(iter);
        ASSERT_EQ(tensor._so, mock_compiled_model._so);
    }
}

TEST_F(AutoLifeTimeTest, loaded_states_bind_buffer) {
    // get Parameter
    config.insert(ov::device::priorities("GPU.0"));
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    config.insert(ov::intel_auto::device_bind_buffer(true));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    auto request = compiled_model->create_infer_request();
    auto states = request->query_state();
    auto res_so = mock_states.front()._so;
    for (auto& state : states)
        ASSERT_EQ(state._so, res_so);
}