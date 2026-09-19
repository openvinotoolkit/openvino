// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/auto/properties.hpp"

using namespace ov::mock_auto_plugin;

namespace {
// Mirrors StatefulModelSupportedTest::create_stateful_model(): a minimal ReadValue/Assign
// pair is enough for filter_device_by_model() to detect the model as stateful.
std::shared_ptr<ov::Model> create_stateful_model() {
    auto arg = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1});
    auto init_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
    const std::string variable_name("variable0");
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{init_const->get_shape(), ov::element::f32, variable_name});
    auto read = std::make_shared<ov::opset11::ReadValue>(init_const, variable);
    auto add = std::make_shared<ov::opset11::Add>(arg, read);
    add->set_friendly_name("add_sum");
    auto assign = std::make_shared<ov::opset11::Assign>(add, variable);
    assign->set_friendly_name("save");
    auto res = std::make_shared<ov::opset11::Result>(add);
    res->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg}));
}
}  // namespace

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

// A stateful model must keep using the classic (non dynamic) schedule even when a resource
// aware selection property is configured: filter_device_by_model() detects the state and
// Plugin::compile_model_impl() must not turn per inference dynamic selection on for it.
TEST_F(AutoDynamicDeviceSelectionTest, stateful_model_disables_dynamic_selection) {
    model = create_stateful_model();
    config.insert(ov::intel_auto::low_power_device("CPU"));
    // optimal_number_of_infer_requests is mocked to 1, which the classic schedule promotes to
    // 2; the dynamic schedule would instead keep a single worker per device.
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request()).Times(2).WillRepeatedly([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
}

// Reproduces try_to_compile_model()'s on-demand fallback: the per inference target (GPU.1)
// fails to compile, so try_to_compile_model() internally re-selects and successfully compiles
// CPU. Note the failing device cannot be CPU itself: try_to_compile_model() deliberately skips
// the fallback re-selection when the device that just failed is CPU (it is already the last
// resort). ensure_device_ready() must release CPU's process-wide priority registration once it
// becomes the active dynamic worker, otherwise CPU stays reserved under our model's priority
// and a lower priority AUTO instance can never select it.
TEST_F(AutoDynamicDeviceSelectionTest, fallback_device_priority_is_released_after_a_successful_retry) {
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));

    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(ov::Throw("mock compile failure"));

    // GPU.0 (the initial ACTUALDEVICE) looks fine while the model is compiled; it is only
    // pushed over its threshold once inferences start, forcing the per inference target off it.
    std::atomic<bool> gpu0_over_threshold{false};
    ON_CALL(*plugin, get_device_utilization)
        .WillByDefault(
            [&gpu0_over_threshold](const std::string& device_name, const std::string&) -> std::optional<float> {
                if (device_name == "GPU.0" && gpu0_over_threshold) {
                    return 95.0f;
                }
                return std::nullopt;
            });

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    gpu0_over_threshold = true;
    run_inferences(compiled_model, 1);

    // A lower priority AUTO instance querying the fallback device must still be able to select
    // it: with the leak, CPU would still be reserved under our HIGH priority model and GPU.1
    // would be picked instead (even though GPU.1's compile attempt actually failed).
    auto verification_devices = plugin->Plugin::parse_meta_devices("CPU,GPU.1", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(
        result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, ""));
    EXPECT_EQ(result.device_name, ov::test::utils::DEVICE_CPU);
    plugin->unregister_priority(2, result.unique_name);
}
