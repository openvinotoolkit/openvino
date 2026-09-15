// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "openvino/runtime/auto/properties.hpp"

using namespace ov::mock_auto_plugin;

// Covers the per inference device selection which AUTO turns on as soon as one of the resource aware
// selection properties is set: the target device is re-selected for every incoming inference, inference is
// serialized per compiled model and every device is given a single worker infer request.
class AutoDynamicDeviceSelectionTest : public tests::AutoTest, public ::testing::Test {
public:
    void SetUp() override {
        plugin->set_device_name("AUTO");
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        config.insert(ov::device::priorities("GPU.0,CPU"));
    }

    void TearDown() override {
        testing::Mock::VerifyAndClearExpectations(core.get());
        testing::Mock::VerifyAndClearExpectations(plugin.get());
    }

    // Pins the device returned by every select_device() call, so that the schedule behavior can be driven
    // from the test instead of depending on the real telemetry backend.
    void expect_selected_devices(const std::vector<std::string>& device_names) {
        ON_CALL(*plugin, select_device)
            .WillByDefault([this, device_names](const std::vector<DeviceInformation>& meta_devices,
                                                const std::string&,
                                                unsigned int,
                                                const ov::auto_plugin::DeviceSelectionPolicy&,
                                                const std::string&) {
                const auto& expected = device_names[m_select_device_count++ % device_names.size()];
                for (const auto& device : meta_devices) {
                    if (device.device_name == expected) {
                        return device;
                    }
                }
                return meta_devices.front();
            });
    }

    void run_inferences(const std::shared_ptr<ov::ICompiledModel>& compiled_model, size_t count) {
        std::shared_ptr<ov::IAsyncInferRequest> infer_request;
        OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
        for (size_t i = 0; i < count; i++) {
            OV_ASSERT_NO_THROW(infer_request->infer());
        }
    }

    size_t m_select_device_count = 0;
};

TEST_F(AutoDynamicDeviceSelectionTest, disabled_by_default_keeps_more_than_one_worker) {
    config.insert(ov::intel_auto::enable_startup_fallback(false));
    // optimal_number_of_infer_requests is mocked to 1, which AUTO promotes to 2 in the classic schedule
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request()).Times(2).WillRepeatedly([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
}

TEST_F(AutoDynamicDeviceSelectionTest, utilization_threshold_forces_single_worker) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request()).Times(1).WillRepeatedly([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
}

TEST_F(AutoDynamicDeviceSelectionTest, low_power_device_forces_single_worker) {
    config.insert(ov::intel_auto::low_power_device("CPU"));
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request()).Times(1).WillRepeatedly([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
}

TEST_F(AutoDynamicDeviceSelectionTest, perf_curve_table_forces_single_worker) {
    config.insert(ov::intel_auto::perf_curve_table(ov::intel_auto::PerfCurveTable{{"iGPU", {{0, 1.0f}, {100, 5.0f}}}}));
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request()).Times(1).WillRepeatedly([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
}

TEST_F(AutoDynamicDeviceSelectionTest, device_is_reselected_for_every_inference) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    expect_selected_devices({"GPU.0"});
    constexpr size_t infer_num = 3;
    // one selection while compiling the model plus one selection per incoming inference
    EXPECT_CALL(*plugin, select_device).Times(1 + infer_num);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, infer_num);
}

TEST_F(AutoDynamicDeviceSelectionTest, staying_on_the_same_device_does_not_recompile) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    expect_selected_devices({"GPU.0"});
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(1);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 3);
}

TEST_F(AutoDynamicDeviceSelectionTest, each_device_is_compiled_only_once_when_switching_back_and_forth) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    expect_selected_devices({"GPU.0", "CPU"});
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(2);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 4);
}

// A single worker per device used to be forbidden because the classic schedule could stall, the execution gate
// makes it safe again, so a long sequence of inferences must keep completing.
TEST_F(AutoDynamicDeviceSelectionTest, repeated_inferences_with_a_single_worker_do_not_stall) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    expect_selected_devices({"GPU.0", "GPU.0", "CPU"});
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 20);
}
