// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

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

// Fails the first start_async() calls the way a device does when it cannot accept a request.
class FailingStartAsyncRequest : public ov::IAsyncInferRequest {
public:
    FailingStartAsyncRequest(const std::shared_ptr<ov::IInferRequest>& request,
                             const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                             const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
                             const std::shared_ptr<std::atomic<int>>& failures)
        : ov::IAsyncInferRequest(request, task_executor, callback_executor),
          m_failures(failures) {}

    void start_async() override {
        if (m_failures->load() > 0) {
            --(*m_failures);
            OPENVINO_THROW("mock start_async failure");
        }
        ov::IAsyncInferRequest::start_async();
    }

private:
    std::shared_ptr<std::atomic<int>> m_failures;
};

// Carries nothing but the name of the device it was created on.
class DeviceTaggedRemoteTensor : public ov::IRemoteTensor {
public:
    DeviceTaggedRemoteTensor(const std::string& device_name, const ov::element::Type& type, const ov::Shape& shape)
        : m_device_name(device_name),
          m_element_type(type),
          m_shape(shape) {}

    const ov::AnyMap& get_properties() const override {
        return m_properties;
    }
    const std::string& get_device_name() const override {
        return m_device_name;
    }
    void set_shape(ov::Shape) override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    const ov::element::Type& get_element_type() const override {
        return m_element_type;
    }
    const ov::Shape& get_shape() const override {
        return m_shape;
    }
    const ov::Strides& get_strides() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    ov::AnyMap m_properties;
    std::string m_device_name;
    ov::element::Type m_element_type;
    ov::Shape m_shape;
};
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
                                                const std::optional<DeviceInformation>&) {
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

    // GPU.0 looks fine until the flag is raised, which forces the per inference target off it.
    void mock_gpu0_over_threshold(const std::shared_ptr<std::atomic<bool>>& over_threshold) {
        ON_CALL(*plugin, get_device_utilizations)
            .WillByDefault([over_threshold](const std::list<DeviceInformation>& devices) {
                std::unordered_map<std::string, float> result;
                if (*over_threshold) {
                    for (const auto& device : devices) {
                        if (device.device_name == "GPU.0") {
                            result[device.device_name] = 95.0f;
                        }
                    }
                }
                return result;
            });
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

TEST_F(AutoDynamicDeviceSelectionTest, fallback_onto_a_cached_device_does_not_compile_it_again) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1"));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(ov::Throw("mock compile failure"));
    // initial device, the device picked for the first inference, then the fallback after GPU.1 fails to compile
    expect_selected_devices({"GPU.0", "GPU.1", "GPU.0"});
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(1);
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(1);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 1);
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

// filter_device_by_model() used to only classify a model as stateful after it had more than one
// candidate device, so a single-candidate stateful model with a resource aware property configured slipped
// through with dynamic selection enabled - breaking the passthrough guarantee that each user created
// infer request gets its own, independent underlying request/state.
TEST_F(AutoDynamicDeviceSelectionTest, stateful_model_with_a_single_candidate_disables_dynamic_selection) {
    // overwrite the GPU.0,CPU priorities inserted by SetUp(): ov::AnyMap::insert() keeps the first entry
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0"));
    model = create_stateful_model();
    // the GPU mock was built around the default model; rebind its ports and sync request to the stateful one
    ON_CALL(*mockIExeNetActual.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mockIExeNetActual.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    inferReqInternalActual = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);
    config.insert(ov::intel_auto::low_power_device("CPU"));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    // select_device() is already called once by init() to pick the sole candidate; if dynamic selection were
    // (incorrectly) left enabled it would be called again for every inference below.
    EXPECT_CALL(*plugin, select_device).Times(0);
    run_inferences(compiled_model, 3);
}

// A stateful model with a single candidate must keep startup/runtime fallback disabled, even though the
// single candidate returns early from filter_device_by_model(). Explicitly enabled fallbacks must not survive.
TEST_F(AutoDynamicDeviceSelectionTest, stateful_single_candidate_disables_startup_and_runtime_fallback) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0"));
    model = create_stateful_model();
    config.insert(ov::intel_auto::low_power_device(ov::test::utils::DEVICE_CPU));
    config.insert(ov::intel_auto::enable_startup_fallback(true));
    config.insert(ov::intel_auto::enable_runtime_fallback(true));
    // the mocked GPU model is rebound to the stateful one, so the passthrough schedule can serve requests
    ON_CALL(*mockIExeNetActual.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mockIExeNetActual.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    inferReqInternalActual = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    // stateful: no per inference selection
    EXPECT_CALL(*plugin, select_device).Times(0);
    run_inferences(compiled_model, 3);
}

// Reproduces try_to_compile_model()'s on-demand fallback: the per inference target (GPU.1)
// fails to compile, so try_to_compile_model() internally re-selects and successfully compiles
// CPU. Note the failing device cannot be CPU itself: try_to_compile_model() deliberately skips
// the fallback re-selection when the device that just failed is CPU (it is already the last
// resort). ensure_device_ready() must drop the registration made for CPU while compiling; the
// model's single persistent registration then follows it to CPU and has to be gone with the model,
// otherwise CPU stays reserved under our model's priority and a lower priority AUTO instance can
// never select it.
TEST_F(AutoDynamicDeviceSelectionTest, fallback_device_priority_is_released_after_a_successful_retry) {
    // overwrite the GPU.0,CPU priorities inserted by SetUp(): ov::AnyMap::insert() keeps the first entry
    config.erase(ov::device::priorities.name());
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
    ON_CALL(*plugin, get_device_utilizations)
        .WillByDefault([&gpu0_over_threshold](const std::list<DeviceInformation>& devices) {
            std::unordered_map<std::string, float> result;
            if (gpu0_over_threshold) {
                for (const auto& device : devices) {
                    if (device.device_name == "GPU.0") {
                        result[device.device_name] = 95.0f;
                    }
                }
            }
            return result;
        });

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    gpu0_over_threshold = true;
    run_inferences(compiled_model, 1);

    // The model now runs on CPU, so CPU stays reserved under our HIGH priority for as long as it is alive.
    auto verification_devices = plugin->Plugin::parse_meta_devices("CPU,GPU.1", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(
        result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, "GPU.1");
    plugin->unregister_priority(2, result.unique_name);

    // With a leaked fallback registration, CPU would stay reserved even after the model is gone.
    compiled_model.reset();
    OV_ASSERT_NO_THROW(
        result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, ov::test::utils::DEVICE_CPU);
    plugin->unregister_priority(2, result.unique_name);
}

// Regression for ensure_device_ready(): unlike the successful retry case above, the fallback
// device's compile can itself fail. try_to_compile_model() bails out immediately without
// unregistering once the device it just attempted is CPU (it is already the last resort, so
// there is nothing left to cascade to), regardless of whether that attempt actually succeeded.
// ensure_device_ready() must still release CPU's process-wide priority registration in that
// case, otherwise CPU stays reserved under our model's priority forever even though nothing
// ever successfully compiled or ran on it.
TEST_F(AutoDynamicDeviceSelectionTest, fallback_device_priority_is_released_after_a_failed_retry) {
    // overwrite the GPU.0,CPU priorities inserted by SetUp(): ov::AnyMap::insert() keeps the first entry
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));

    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(ov::Throw("mock compile failure on GPU.1"));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          _))
        .WillByDefault(ov::Throw("mock compile failure on CPU"));

    // GPU.0 (the initial ACTUALDEVICE) looks fine while the model is compiled; it is only
    // pushed over its threshold once inferences start, forcing the per inference target off it.
    std::atomic<bool> gpu0_over_threshold{false};
    ON_CALL(*plugin, get_device_utilizations)
        .WillByDefault([&gpu0_over_threshold](const std::list<DeviceInformation>& devices) {
            std::unordered_map<std::string, float> result;
            if (gpu0_over_threshold) {
                for (const auto& device : devices) {
                    if (device.device_name == "GPU.0") {
                        result[device.device_name] = 95.0f;
                    }
                }
            }
            return result;
        });

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    gpu0_over_threshold = true;
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    EXPECT_THROW(infer_request->infer(), ov::Exception);

    // A lower priority AUTO instance querying CPU must still be able to select it: with the
    // leak, CPU would still be reserved under our HIGH priority model even though nothing
    // actually compiled or is running on it.
    auto verification_devices = plugin->Plugin::parse_meta_devices("CPU", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(
        result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, ov::test::utils::DEVICE_CPU);
    plugin->unregister_priority(2, result.unique_name);
}

// Regression for try_to_compile_model(): the configured low_power_device is a standing override
// that is force-selected even when it is not part of the configured device priorities (see
// SelectDeviceWithLowPowerDevicePrecedenceTest::lowPowerDeviceSelectedEvenWhenNotInCandidateList).
// If that forced device fails to compile, try_to_compile_model() used to reselect with the same
// low_power_device argument, which forced the very same failing device back in and recursed on it
// forever instead of falling back to the configured device priorities.
TEST_F(AutoDynamicDeviceSelectionTest, low_power_device_failure_falls_back_instead_of_looping_forever) {
    // GPU.1 is never part of the configured priorities ("GPU.0,CPU" from SetUp()), so it can only
    // be selected through the low_power_device override.
    config.insert(ov::intel_auto::low_power_device("GPU.1"));
    ON_CALL(*plugin, get_low_power_mode()).WillByDefault(Return(true));
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                              _))
        .Times(1)
        .WillOnce(ov::Throw("mock compile failure on GPU.1"));
    // once GPU.1 is excluded, compilation must fall back to the configured device priorities (GPU.0)
    // instead of forcing GPU.1 again and recursing forever.
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                              _))
        .Times(1)
        .WillOnce(Return(mockExeNetworkActual));
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request()).WillRepeatedly([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 1);
}

// Reproduces schedule_dynamic_task()'s queueing path: while one dynamic inference is still
// being dispatched (the execution gate is busy), a second inference started concurrently must
// be queued in m_gate_pending_tasks and only dispatched once release_execution_slot() picks it
// up, instead of being dropped or racing the in-flight one. None of the other tests in this
// file exercise this because they all issue requests through the synchronous infer() one at a
// time via run_inferences().
TEST_F(AutoDynamicDeviceSelectionTest, concurrent_requests_are_queued_and_dispatched_in_turn) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));

    std::mutex mtx;
    std::condition_variable cv;
    bool first_inference_is_blocked = false;
    bool release_first_inference = false;

    ON_CALL(*plugin, select_device)
        .WillByDefault([this, &mtx, &cv, &first_inference_is_blocked, &release_first_inference](
                           const std::vector<DeviceInformation>& meta_devices,
                           const std::string&,
                           unsigned int,
                           const ov::auto_plugin::DeviceSelectionPolicy&,
                           const std::optional<DeviceInformation>&) {
            // call #0 happens during compile_model(); block only the first per inference
            // reselection (call #1) so the second start_async() below is guaranteed to observe
            // the execution gate as busy and go through the queueing path.
            if (m_select_device_count++ == 1) {
                std::unique_lock<std::mutex> lock(mtx);
                first_inference_is_blocked = true;
                cv.notify_all();
                cv.wait(lock, [&release_first_inference] {
                    return release_first_inference;
                });
            }
            for (const auto& device : meta_devices) {
                if (device.device_name == "GPU.0") {
                    return device;
                }
            }
            return meta_devices.front();
        });

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    std::shared_ptr<ov::IAsyncInferRequest> first_request;
    std::shared_ptr<ov::IAsyncInferRequest> second_request;
    OV_ASSERT_NO_THROW(first_request = compiled_model->create_infer_request());
    OV_ASSERT_NO_THROW(second_request = compiled_model->create_infer_request());

    OV_ASSERT_NO_THROW(first_request->start_async());
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&first_inference_is_blocked] {
            return first_inference_is_blocked;
        });
    }

    // the execution gate is held by the still-dispatching first request, so this must be
    // queued in m_gate_pending_tasks rather than dispatched (or dropped) right away.
    OV_ASSERT_NO_THROW(second_request->start_async());
    EXPECT_EQ(m_select_device_count, 2u);

    {
        std::lock_guard<std::mutex> lock(mtx);
        release_first_inference = true;
    }
    cv.notify_all();

    OV_ASSERT_NO_THROW(first_request->wait());
    OV_ASSERT_NO_THROW(second_request->wait());
    EXPECT_EQ(m_select_device_count, 3u);
}

// The low_power_device override must be dropped once its on-demand compilation fails, otherwise every
// following inference keeps selecting the failed device instead of the working one.
TEST_F(AutoDynamicDeviceSelectionTest, low_power_device_compile_failure_disables_the_override) {
    config.insert(ov::intel_auto::low_power_device(ov::test::utils::DEVICE_CPU));
    auto low_power_mode = std::make_shared<std::atomic<bool>>(false);
    ON_CALL(*plugin, get_low_power_mode()).WillByDefault([low_power_mode]() {
        return std::optional<bool>(low_power_mode->load());
    });
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          _))
        .WillByDefault(ov::Throw("mock compile failure on CPU"));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    low_power_mode->store(true);
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    EXPECT_THROW(infer_request->infer(), ov::Exception);
    OV_ASSERT_NO_THROW(infer_request->infer());
}

// A failed on-demand compilation must only release the priority registration it owns: the one of another
// AUTO model with the same priority on that device has to survive.
TEST_F(AutoDynamicDeviceSelectionTest, failed_on_demand_compile_keeps_the_priority_of_another_model) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(ov::Throw("mock compile failure"));
    auto gpu0_over_threshold = std::make_shared<std::atomic<bool>>(false);
    mock_gpu0_over_threshold(gpu0_over_threshold);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    // another AUTO model with the same priority is running on GPU.1
    const auto gpu1 = plugin->Plugin::parse_meta_devices("GPU.1", config).front();
    plugin->register_priority(0, gpu1.unique_name);

    gpu0_over_threshold->store(true);
    run_inferences(compiled_model, 1);

    // GPU.1 is still reserved by the other model, so a lower priority instance must not get it
    auto verification_devices = plugin->Plugin::parse_meta_devices("GPU.1,CPU", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, ov::test::utils::DEVICE_CPU);
    plugin->unregister_priority(2, result.unique_name);
    plugin->unregister_priority(0, gpu1.unique_name);
}

// AUTO/MULTI as low_power_device would make the plugin compile itself again and again.
TEST_F(AutoDynamicDeviceSelectionTest, meta_device_as_low_power_device_is_rejected) {
    ON_CALL(*plugin, is_meta_device).WillByDefault([this](const std::string& priorities) {
        return plugin->Plugin::is_meta_device(priorities);
    });
    for (const auto* meta_device : {"AUTO", "MULTI:CPU"}) {
        auto properties = config;
        properties.insert(ov::intel_auto::low_power_device(meta_device));
        EXPECT_THROW(plugin->compile_model(model, properties), ov::Exception) << meta_device;
    }
}

TEST_F(AutoDynamicDeviceSelectionTest, explicit_startup_fallback_is_rejected) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::intel_auto::enable_startup_fallback(true));
    EXPECT_THROW(plugin->compile_model(model, config), ov::Exception);
}

// With a single candidate there is nothing to switch to, so the passthrough schedule must be kept.
TEST_F(AutoDynamicDeviceSelectionTest, single_candidate_device_disables_dynamic_selection) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0"));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    // only the selection done while compiling the model, none per inference
    EXPECT_CALL(*plugin, select_device).Times(1);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 3);
}

// A low power device out of the candidate list is still something to switch to.
TEST_F(AutoDynamicDeviceSelectionTest, low_power_device_out_of_a_single_candidate_enables_dynamic_selection) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0"));
    config.insert(ov::intel_auto::low_power_device(ov::test::utils::DEVICE_CPU));
    constexpr size_t infer_num = 3;
    EXPECT_CALL(*plugin, select_device).Times(1 + infer_num);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, infer_num);
}

// The per inference selection serializes the inferences, so THROUGHPUT is not honored and the properties are ignored.
TEST_F(AutoDynamicDeviceSelectionTest, throughput_hint_disables_dynamic_selection) {
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    config.insert(ov::intel_auto::enable_startup_fallback(false));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    EXPECT_CALL(*plugin, select_device).Times(1);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 3);
}

TEST_F(AutoDynamicDeviceSelectionTest, optimal_number_of_infer_requests_is_one) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(Return(ov::Any(4u)));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_EQ(compiled_model->get_property(ov::optimal_number_of_infer_requests.name()).as<unsigned int>(), 1u);
}

// The only worker of a device must be given back when its request cannot be started, otherwise the execution
// gate stays closed and every following inference is queued forever.
TEST_F(AutoDynamicDeviceSelectionTest, failure_to_start_a_request_does_not_block_following_inferences) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    auto failures = std::make_shared<std::atomic<int>>(1);
    ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault([this, failures]() {
        return std::static_pointer_cast<ov::IAsyncInferRequest>(
            std::make_shared<FailingStartAsyncRequest>(inferReqInternalActual,
                                                       std::make_shared<ov::threading::ImmediateExecutor>(),
                                                       nullptr,
                                                       failures));
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    std::shared_ptr<ov::IAsyncInferRequest> failed_request;
    OV_ASSERT_NO_THROW(failed_request = compiled_model->create_infer_request());
    EXPECT_THROW(failed_request->infer(), ov::Exception);

    std::shared_ptr<ov::IAsyncInferRequest> next_request;
    OV_ASSERT_NO_THROW(next_request = compiled_model->create_infer_request());
    OV_ASSERT_NO_THROW(next_request->start_async());
    EXPECT_TRUE(next_request->wait_for(std::chrono::seconds(10)));
}

// MODEL_PRIORITY reserves the device the model currently runs on: once the per inference selection moves
// to another device, the initial device is free again for lower priority AUTO instances.
TEST_F(AutoDynamicDeviceSelectionTest, priority_registration_follows_the_active_device) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(Return(mockExeNetworkActual));
    auto gpu0_over_threshold = std::make_shared<std::atomic<bool>>(false);
    mock_gpu0_over_threshold(gpu0_over_threshold);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    gpu0_over_threshold->store(true);
    run_inferences(compiled_model, 1);

    // the model moved to GPU.1, so a lower priority instance has to be steered to GPU.0 instead
    auto verification_devices = plugin->Plugin::parse_meta_devices("GPU.0,GPU.1", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, "GPU.0");
    plugin->unregister_priority(2, result.unique_name);
}

TEST_F(AutoDynamicDeviceSelectionTest, priority_registration_is_released_on_destruction) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(Return(mockExeNetworkActual));
    auto gpu0_over_threshold = std::make_shared<std::atomic<bool>>(false);
    mock_gpu0_over_threshold(gpu0_over_threshold);

    {
        std::shared_ptr<ov::ICompiledModel> compiled_model;
        OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
        gpu0_over_threshold->store(true);
        run_inferences(compiled_model, 1);
    }

    // neither the initial device nor the one the model switched to may stay reserved afterwards
    auto verification_devices = plugin->Plugin::parse_meta_devices("GPU.1,GPU.0", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, "GPU.1");
    plugin->unregister_priority(2, result.unique_name);
}

// Every candidate fails to compile in turn, so the cascade runs out of devices: the registration of the last one
// must be released once, not twice, or the model's own registration of that device is taken away.
TEST_F(AutoDynamicDeviceSelectionTest, exhausted_compile_cascade_releases_each_registration_once) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,GPU.2"));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    auto gpu0_compile_fails = std::make_shared<std::atomic<bool>>(false);
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                          _))
        .WillByDefault([this, gpu0_compile_fails](const std::shared_ptr<const ov::Model>&,
                                                  const std::string&,
                                                  const ov::AnyMap&) -> ov::SoPtr<ov::ICompiledModel> {
            if (gpu0_compile_fails->load()) {
                OPENVINO_THROW("mock compile failure on GPU.0");
            }
            return mockExeNetworkActual;
        });
    for (const auto* device : {"GPU.1", "GPU.2"}) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(device)),
                              _))
            .WillByDefault(ov::Throw("mock compile failure"));
    }
    auto gpu0_over_threshold = std::make_shared<std::atomic<bool>>(false);
    mock_gpu0_over_threshold(gpu0_over_threshold);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    gpu0_compile_fails->store(true);
    gpu0_over_threshold->store(true);
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    // GPU.1 and GPU.2 fail to compile, the cascade ends on the cached GPU.0 which must not be compiled again
    OV_ASSERT_NO_THROW(infer_request->infer());

    // the model still runs on GPU.0, so a lower priority instance must keep getting GPU.1
    auto verification_devices = plugin->Plugin::parse_meta_devices("GPU.0,GPU.1", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, "GPU.1");
    plugin->unregister_priority(2, result.unique_name);
}

TEST_F(AutoDynamicDeviceSelectionTest, execution_devices_follows_the_active_device) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(Return(mockExeNetworkActual));
    auto gpu0_over_threshold = std::make_shared<std::atomic<bool>>(false);
    mock_gpu0_over_threshold(gpu0_over_threshold);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    const auto execution_devices = [&compiled_model] {
        return compiled_model->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
    };
    EXPECT_EQ(execution_devices(), std::vector<std::string>{"GPU.0"});

    gpu0_over_threshold->store(true);
    run_inferences(compiled_model, 1);
    EXPECT_EQ(execution_devices(), std::vector<std::string>{"GPU.1"});
}

// The low power device may sit outside the candidates: a tensor created on it still has to pin the request there.
TEST_F(AutoDynamicDeviceSelectionTest, remote_tensor_of_an_out_of_list_low_power_device_is_accepted) {
    config.insert(ov::intel_auto::low_power_device("GPU.1"));
    ON_CALL(*plugin, get_low_power_mode()).WillByDefault(Return(true));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(Return(mockExeNetworkActual));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    const auto input = compiled_model->inputs().at(0);
    auto remote_tensor = std::make_shared<DeviceTaggedRemoteTensor>("GPU.1", input.get_element_type(), input.get_shape());
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    OV_ASSERT_NO_THROW(infer_request->set_tensor(input, ov::SoPtr<ov::ITensor>{remote_tensor, nullptr}));
    OV_ASSERT_NO_THROW(infer_request->start_async());
    OV_ASSERT_NO_THROW(infer_request->wait());
}

// A remote output tensor pins the request to its device, the same way a remote input does.
TEST_F(AutoDynamicDeviceSelectionTest, remote_output_tensor_pins_the_request_to_its_device) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1"));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(Return(mockExeNetworkActual));
    // only the selection done while compiling the model: a pinned request skips the per inference selection
    EXPECT_CALL(*plugin, select_device).Times(1);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    const auto output = compiled_model->outputs().at(0);
    auto remote_tensor =
        std::make_shared<DeviceTaggedRemoteTensor>("GPU.1", output.get_element_type(), output.get_shape());
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    OV_ASSERT_NO_THROW(infer_request->set_tensor(output, ov::SoPtr<ov::ITensor>{remote_tensor, nullptr}));
    OV_ASSERT_NO_THROW(infer_request->start_async());
    OV_ASSERT_NO_THROW(infer_request->wait());
}

// The request can only run on one device, so remote tensors from different devices cannot be mixed.
TEST_F(AutoDynamicDeviceSelectionTest, remote_input_and_output_tensors_of_different_devices_are_rejected) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1"));
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    const auto input = compiled_model->inputs().at(0);
    const auto output = compiled_model->outputs().at(0);
    auto remote_input =
        std::make_shared<DeviceTaggedRemoteTensor>("GPU.0", input.get_element_type(), input.get_shape());
    auto remote_output =
        std::make_shared<DeviceTaggedRemoteTensor>("GPU.1", output.get_element_type(), output.get_shape());
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    OV_ASSERT_NO_THROW(infer_request->set_tensor(input, ov::SoPtr<ov::ITensor>{remote_input, nullptr}));
    OV_ASSERT_NO_THROW(infer_request->set_tensor(output, ov::SoPtr<ov::ITensor>{remote_output, nullptr}));
    OV_ASSERT_NO_THROW(infer_request->start_async());
    EXPECT_THROW(infer_request->wait(), ov::Exception);
}
