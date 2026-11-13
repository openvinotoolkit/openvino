// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/compilation_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

using namespace ov::mock_auto_plugin;

namespace ov::test {
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

// Completes the callback task in place, then holds the calling thread, so that the code which follows the completion
// of a user request on the device thread can be paused.
class HoldingCallbackExecutor : public ov::threading::ITaskExecutor {
public:
    void run(ov::threading::Task task) override {
        task();
        std::unique_lock<std::mutex> lock(m_mutex);
        m_held = true;
        m_cv.notify_all();
        m_cv.wait(lock, [this] {
            return m_released;
        });
    }

    bool wait_until_held(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_cv.wait_for(lock, timeout, [this] {
            return m_held;
        });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_released = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_held = false;
    bool m_released = false;
};

// set_callback_executor() is protected in ov::ICompiledModel; it is reached through a pointer to member.
struct CallbackExecutorAccess : public ov::ICompiledModel {
    using ov::ICompiledModel::set_callback_executor;
};

// Runs every task on its own thread, so a device completion callback is not issued from the dispatching thread.
class ThreadPerTaskExecutor : public ov::threading::ITaskExecutor {
public:
    void run(ov::threading::Task task) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_threads.emplace_back(std::move(task));
    }

    ~ThreadPerTaskExecutor() override {
        for (auto& thread : m_threads) {
            if (thread.get_id() == std::this_thread::get_id()) {
                thread.detach();
            } else if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    std::mutex m_mutex;
    std::vector<std::thread> m_threads;
};

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

    // The per inference selection takes no part in the model priority arbitration: no device stays reserved, so a
    // lower priority instance can pick any of them.
    void expect_no_device_reserved(const std::vector<std::string>& device_names) {
        ASSERT_GT(device_names.size(), 1u);
        ov::auto_plugin::DeviceSelectionPolicy empty_policy;
        for (size_t i = 0; i < device_names.size(); i++) {
            const auto& device_name = device_names[i];
            const auto& other_name = device_names[(i + 1) % device_names.size()];
            auto devices = plugin->Plugin::parse_meta_devices(device_name + "," + other_name, config);
            DeviceInformation result;
            OV_ASSERT_NO_THROW(result = plugin->Plugin::select_device(devices, "FP32", 2, empty_policy, {}));
            EXPECT_EQ(result.device_name, device_name);
            plugin->unregister_priority(2, result.unique_name);
        }
    }

    std::atomic<size_t> m_select_device_count{0};
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
    // the mocked GPU model is rebound to the stateful one, so the passthrough schedule can serve requests
    ON_CALL(*mockIExeNetActual.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mockIExeNetActual.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    inferReqInternalActual = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    // the dynamic schedule would select a device again for every inference
    EXPECT_CALL(*plugin, select_device).Times(0);
    run_inferences(compiled_model, 3);
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

// Regression for ensure_device_ready(): the fallback device chosen while compiling on demand must not stay reserved.
TEST_F(AutoDynamicDeviceSelectionTest, no_device_is_reserved_after_a_successful_fallback) {
    // overwrite the GPU.0,CPU priorities inserted by SetUp(): ov::AnyMap::insert() keeps the first entry
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
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
    gpu0_over_threshold->store(true);
    run_inferences(compiled_model, 1);
    expect_no_device_reserved({"GPU.0", "GPU.1", ov::test::utils::DEVICE_CPU});

    compiled_model.reset();
    expect_no_device_reserved({"GPU.0", "GPU.1", ov::test::utils::DEVICE_CPU});
}

// Regression for ensure_device_ready(): a fallback device whose compile fails too must not stay reserved either.
TEST_F(AutoDynamicDeviceSelectionTest, no_device_is_reserved_after_a_failed_fallback) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
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
    auto gpu0_over_threshold = std::make_shared<std::atomic<bool>>(false);
    mock_gpu0_over_threshold(gpu0_over_threshold);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    gpu0_over_threshold->store(true);
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = compiled_model->create_infer_request());
    EXPECT_THROW(infer_request->infer(), ov::Exception);
    expect_no_device_reserved({"GPU.0", "GPU.1", ov::test::utils::DEVICE_CPU});
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
    EXPECT_EQ(m_select_device_count.load(), 2u);

    {
        std::lock_guard<std::mutex> lock(mtx);
        release_first_inference = true;
    }
    cv.notify_all();

    OV_ASSERT_NO_THROW(first_request->wait());
    OV_ASSERT_NO_THROW(second_request->wait());
    EXPECT_EQ(m_select_device_count.load(), 3u);
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

// The reservation another model holds on the failing device must survive the failed on-demand compile.
TEST_F(AutoDynamicDeviceSelectionTest, failed_on_demand_compile_keeps_the_reservation_of_another_model) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
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

    // another AUTO model with the default (MEDIUM) priority is running on GPU.1
    const auto gpu1 = plugin->Plugin::parse_meta_devices("GPU.1", config).front();
    plugin->register_priority(1, gpu1.unique_name);

    gpu0_over_threshold->store(true);
    run_inferences(compiled_model, 1);

    // GPU.1 is still reserved by the other model, so a lower priority instance must not get it
    auto verification_devices = plugin->Plugin::parse_meta_devices("GPU.1,CPU", config);
    ov::auto_plugin::DeviceSelectionPolicy empty_policy;
    DeviceInformation result;
    OV_ASSERT_NO_THROW(result = plugin->Plugin::select_device(verification_devices, "FP32", 2, empty_policy, {}));
    EXPECT_EQ(result.device_name, ov::test::utils::DEVICE_CPU);
    plugin->unregister_priority(2, result.unique_name);
    plugin->unregister_priority(1, gpu1.unique_name);
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

// Without the telemetry backend the utilization cannot be evaluated, so the properties are ignored instead of
// serializing the inference for nothing.
TEST_F(AutoDynamicDeviceSelectionTest, resource_aware_properties_are_ignored_without_telemetry) {
    ON_CALL(*plugin, is_telemetry_available()).WillByDefault(Return(false));
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

// The device completion callback keeps using the scheduler (it gives the execution slot back) after it has completed
// the user request, so the scheduler must not be destroyed before that callback is out of it.
TEST_F(AutoDynamicDeviceSelectionTest, destruction_waits_for_the_device_completion_callback) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    auto device_executor = std::make_shared<ThreadPerTaskExecutor>();
    auto failures = std::make_shared<std::atomic<int>>(0);
    ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault([this, device_executor, failures]() {
        return std::static_pointer_cast<ov::IAsyncInferRequest>(
            std::make_shared<FailingStartAsyncRequest>(inferReqInternalActual, device_executor, nullptr, failures));
    });
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));

    // the user request is completed on the device thread, which is then held before it releases the execution slot
    auto callback_executor = std::make_shared<HoldingCallbackExecutor>();
    (compiled_model.get()->*(&CallbackExecutorAccess::set_callback_executor))(callback_executor);
    std::shared_ptr<ov::IAsyncInferRequest> request;
    OV_ASSERT_NO_THROW(request = compiled_model->create_infer_request());
    OV_ASSERT_NO_THROW(request->start_async());
    ASSERT_TRUE(callback_executor->wait_until_held(std::chrono::seconds(10)));
    ASSERT_TRUE(request->wait_for(std::chrono::seconds(10)));

    auto teardown_started = std::make_shared<std::promise<void>>();
    auto teardown_done = std::make_shared<std::promise<void>>();
    auto started = teardown_started->get_future();
    auto finished = teardown_done->get_future();
    std::thread teardown([teardown_started,
                          teardown_done,
                          request = std::move(request),
                          compiled_model = std::move(compiled_model)]() mutable {
        teardown_started->set_value();
        request.reset();
        compiled_model.reset();
        teardown_done->set_value();
    });
    ASSERT_EQ(started.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    // destruction has to wait for the held device callback
    EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);

    callback_executor->release();
    if (finished.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        teardown.detach();
        FAIL() << "destruction did not finish after the device completion callback was released";
    }
    teardown.join();
}

// The selection which reserves nothing must not be steered by the reservations of other models either.
TEST_F(AutoDynamicDeviceSelectionTest, reservations_of_other_models_do_not_affect_the_selection) {
    auto devices = plugin->Plugin::parse_meta_devices("GPU.0," + std::string(ov::test::utils::DEVICE_CPU), config);
    const auto gpu0 = devices.front();
    // a model of a higher priority holds GPU.0
    plugin->register_priority(0, gpu0.unique_name);
    ov::auto_plugin::DeviceSelectionPolicy policy;
    DeviceInformation reserving;
    OV_ASSERT_NO_THROW(reserving = plugin->Plugin::select_device(devices, "FP32", 2, policy, {}));
    plugin->unregister_priority(2, reserving.unique_name);
    EXPECT_EQ(reserving.device_name, ov::test::utils::DEVICE_CPU);

    policy.reserve_device = false;
    DeviceInformation ignoring;
    OV_ASSERT_NO_THROW(ignoring = plugin->Plugin::select_device(devices, "FP32", 2, policy, {}));
    EXPECT_EQ(ignoring.device_name, "GPU.0");
    plugin->unregister_priority(0, gpu0.unique_name);
}

TEST_F(AutoDynamicDeviceSelectionTest, no_device_is_reserved_while_switching_devices) {
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
    expect_no_device_reserved({"GPU.0", "GPU.1"});
    gpu0_over_threshold->store(true);
    run_inferences(compiled_model, 1);
    expect_no_device_reserved({"GPU.0", "GPU.1"});
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

// Per inference selection is off here (THROUGHPUT), so the legacy runtime fallback handles the failure. The low power
// device sits outside the candidates, so its very first failure must still fall back to the configured priorities.
TEST_F(AutoDynamicDeviceSelectionTest, out_of_list_low_power_device_first_failure_falls_back_at_runtime) {
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    config.insert(ov::intel_auto::enable_startup_fallback(false));
    config.insert(ov::intel_auto::enable_runtime_fallback(true));
    config.insert(ov::intel_auto::low_power_device("GPU.1"));
    ON_CALL(*plugin, get_low_power_mode()).WillByDefault(Return(true));

    auto mock_gpu1 = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, plugin);
    ON_CALL(*mock_gpu1, inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mock_gpu1, outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    ON_CALL(*mock_gpu1, get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(Return(optimalNum));
    auto executor = std::make_shared<ov::threading::ImmediateExecutor>();
    // a weak reference avoids a cycle through the mock's own action, which gmock would report as a leak
    ON_CALL(*mock_gpu1, create_infer_request()).WillByDefault([weak_gpu1 = std::weak_ptr(mock_gpu1), executor]() {
        auto sync_request = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(weak_gpu1.lock());
        return std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(sync_request, executor, nullptr, true);
    });
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault(Return(ov::SoPtr<ov::ICompiledModel>{mock_gpu1, {}}));
    ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault([this]() {
        return mockIExeNetActual->ICompiledModel::create_infer_request();
    });

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    auto infer_request = compiled_model->create_infer_request();
    OV_ASSERT_NO_THROW(infer_request->infer());

    // destroying the model must release the reservation of the fallback device as well
    infer_request.reset();
    compiled_model.reset();
    expect_no_device_reserved({"GPU.0", "GPU.1"});
}

// Every candidate fails to compile in turn, so the cascade runs out of devices and ends on the cached GPU.0.
TEST_F(AutoDynamicDeviceSelectionTest, exhausted_compile_cascade_reserves_no_device) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,GPU.1,GPU.2"));
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
    // every device is compiled exactly once, GPU.0 when the model is loaded and the failed ones on demand
    for (const auto* device : {"GPU.0", "GPU.1", "GPU.2"}) {
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(device)),
                                  ::testing::Matcher<const ov::AnyMap&>(_)))
            .Times(1);
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
    // both failed devices are excluded, so neither of them is compiled again
    OV_ASSERT_NO_THROW(infer_request->infer());
    expect_no_device_reserved({"GPU.0", "GPU.1", "GPU.2"});
}

TEST_F(AutoDynamicDeviceSelectionTest, explicit_runtime_fallback_is_rejected) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::intel_auto::enable_runtime_fallback(true));
    EXPECT_THROW(plugin->compile_model(model, config), ov::Exception);
}

TEST_F(AutoDynamicDeviceSelectionTest, disabled_runtime_fallback_is_accepted) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::intel_auto::enable_runtime_fallback(false));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 1);
}

TEST_F(AutoDynamicDeviceSelectionTest, explicit_model_priority_is_ignored) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 1);
}

TEST_F(AutoDynamicDeviceSelectionTest, more_than_one_stream_is_rejected) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::num_streams(2));
    EXPECT_THROW(plugin->compile_model(model, config), ov::Exception);
}

TEST_F(AutoDynamicDeviceSelectionTest, single_stream_is_accepted) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::num_streams(1));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 1);
}

// Without the per inference selection these properties keep their usual meaning.
TEST_F(AutoDynamicDeviceSelectionTest, fallback_priority_and_streams_are_accepted_without_resource_aware_properties) {
    config.insert(ov::intel_auto::enable_runtime_fallback(true));
    config.insert(ov::hint::model_priority(ov::hint::Priority::HIGH));
    config.insert(ov::num_streams(2));
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
}

// A failed inference is reported to the caller, it is not retried on another device and it does not block the gate.
TEST_F(AutoDynamicDeviceSelectionTest, inference_failure_is_reported_instead_of_retried_on_another_device) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    auto executor = std::make_shared<ov::threading::ImmediateExecutor>();
    ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault([this, executor]() {
        auto sync_request = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);
        return std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(sync_request, executor, nullptr, true);
    });
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                              _))
        .WillRepeatedly(Return(mockExeNetworkActual));
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
        .Times(0);

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    auto infer_request = compiled_model->create_infer_request();
    EXPECT_THROW(infer_request->infer(), ov::Exception);
    EXPECT_THROW(infer_request->infer(), ov::Exception);
}

// Records the NUM_STREAMS every compile_model() call of the device received, -100 if it was not passed at all.
class StreamsRecorder {
public:
    void attach(tests::AutoTest& test, const std::string& device, const ov::SoPtr<ov::ICompiledModel>& compiled_model) {
        ON_CALL(*test.core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(device)),
                              _))
            .WillByDefault([this, device, compiled_model](const std::shared_ptr<const ov::Model>&,
                                                          const std::string&,
                                                          const ov::AnyMap& properties) {
                const auto iter = properties.find(ov::num_streams.name());
                m_streams[device] = iter == properties.end() ? -100 : iter->second.as<ov::streams::Num>().num;
                return compiled_model;
            });
    }
    int32_t streams(const std::string& device) const {
        const auto iter = m_streams.find(device);
        return iter == m_streams.end() ? -200 : iter->second;
    }

private:
    std::map<std::string, int32_t> m_streams;
};

TEST_F(AutoDynamicDeviceSelectionTest, one_stream_is_used_when_the_user_did_not_choose) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    StreamsRecorder recorder;
    recorder.attach(*this, "GPU.0", mockExeNetworkActual);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_EQ(recorder.streams("GPU.0"), 1);
}

TEST_F(AutoDynamicDeviceSelectionTest, one_stream_is_used_for_the_out_of_list_low_power_device) {
    config.insert(ov::intel_auto::low_power_device("GPU.1"));
    ON_CALL(*plugin, get_low_power_mode()).WillByDefault(Return(true));
    StreamsRecorder recorder;
    recorder.attach(*this, "GPU.1", mockExeNetworkActual);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_EQ(recorder.streams("GPU.1"), 1);
}

TEST_F(AutoDynamicDeviceSelectionTest, explicit_streams_setting_is_kept) {
    config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
    config.insert(ov::num_streams(ov::streams::AUTO));
    StreamsRecorder recorder;
    recorder.attach(*this, "GPU.0", mockExeNetworkActual);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_EQ(recorder.streams("GPU.0"), ov::streams::AUTO.num);
}

TEST_F(AutoDynamicDeviceSelectionTest, streams_are_not_forced_without_the_per_inference_selection) {
    StreamsRecorder recorder;
    recorder.attach(*this, "GPU.0", mockExeNetworkActual);
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_EQ(recorder.streams("GPU.0"), -100);
}

class AutoDynamicInactiveDeviceReleaseTest : public AutoDynamicDeviceSelectionTest {
public:
    void SetUp() override {
        AutoDynamicDeviceSelectionTest::SetUp();
        m_cache_dir = std::filesystem::temp_directory_path() /
                      ("ov_auto_inactive_device_release_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(m_cache_dir);
        ON_CALL(*core, get_property(_, StrEq(ov::cache_dir.name()), _))
            .WillByDefault(Return(ov::Any(m_cache_dir.string())));
        ON_CALL(*core, get_property(_, StrEq(ov::cache_path.name()), _))
            .WillByDefault(Return(ov::Any(std::filesystem::path())));
        ON_CALL(*core, create_compile_config(_, _)).WillByDefault([this](const std::string& device, const ov::AnyMap&) {
            if (device == "CPU") {
                ++m_cpu_cache_key_compute_count;
            }
            return ov::AnyMap{};
        });
        ON_CALL(*core, device_supports_model_caching(_)).WillByDefault(Return(true));
        config.erase(ov::device::priorities.name());
        config.insert(ov::device::priorities("GPU.0,GPU.1,CPU"));
        config.insert(ov::intel_auto::devices_utilization_threshold(std::map<std::string, unsigned>{{"GPU.0", 80}}));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                              _))
            .WillByDefault([this](const std::shared_ptr<const ov::Model>& compiled_model,
                                  const std::string&,
                                  const ov::AnyMap& compile_config) {
                ++m_gpu0_compile_count;
                if (m_write_gpu0_cache_blob) {
                    write_cache_blob("GPU.0", compiled_model, compile_config);
                }
                return mockExeNetworkActual;
            });
        // GPU.1 is only a precompile candidate, but precompile dereferences the compiled model it gets back
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault([this](const std::shared_ptr<const ov::Model>& compiled_model,
                                  const std::string&,
                                  const ov::AnyMap& compile_config) {
                ++m_cpu_compile_count;
                // what was compiled for CPU before must be gone by now, nothing but AUTO holds it
                EXPECT_TRUE(m_cpu_model.expired());
                EXPECT_TRUE(m_cpu_worker_request.expired());
                if (m_write_cpu_cache_blob) {
                    write_cache_blob("CPU", compiled_model, compile_config);
                }
                return make_tracked_cpu_model();
            });
    }

    void TearDown() override {
        AutoDynamicDeviceSelectionTest::TearDown();
        std::error_code error;
        std::filesystem::remove_all(m_cache_dir, error);
    }

    void write_cache_blob(const std::string& device,
                          const std::shared_ptr<const ov::Model>& compiled_model,
                          const ov::AnyMap& compile_config) {
        const auto supported_config = core->get_supported_property(device, compile_config, true);
        const auto cache_key_config = core->create_compile_config(device, supported_config);
        const auto blob_id = ov::ModelCache::compute_hash(compiled_model, cache_key_config);
        std::ofstream(m_cache_dir / (blob_id + ".blob"), std::ios::binary).put('\0');
    }

    ov::SoPtr<ov::ICompiledModel> make_tracked_cpu_model() {
        auto cpu_model = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, mock_plugin_cpu);
        const std::weak_ptr<ov::MockICompiledModel> weak_model = cpu_model;
        ON_CALL(*cpu_model, inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
        ON_CALL(*cpu_model, outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
        ON_CALL(*cpu_model, get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
            .WillByDefault(Return(optimalNum));
        ON_CALL(*cpu_model, get_property(StrEq(ov::supported_properties.name())))
            .WillByDefault(Return(std::vector<ov::PropertyName>{ov::model_name, ov::loaded_from_cache}));
        ON_CALL(*cpu_model, get_property(StrEq(ov::model_name.name()))).WillByDefault(Return("cpu_model"));
        ON_CALL(*cpu_model, get_property(StrEq(ov::loaded_from_cache.name()))).WillByDefault(Return(true));
        ON_CALL(*cpu_model, get_runtime_model()).WillByDefault(Return(model));
        ON_CALL(*cpu_model, create_sync_infer_request()).WillByDefault([this, weak_model]() {
            auto request = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(weak_model.lock());
            m_cpu_worker_request = request;
            if (m_cpu_profiling_info) {
                ON_CALL(*request, get_profiling_info()).WillByDefault([this]() {
                    return m_cpu_profiling_info();
                });
            }
            return request;
        });
        ON_CALL(*cpu_model, create_infer_request()).WillByDefault([weak_model]() {
            return weak_model.lock()->ICompiledModel::create_infer_request();
        });
        m_cpu_model = cpu_model;
        return ov::SoPtr<ov::ICompiledModel>{cpu_model, {}};
    }

    // GPU.0 is the initial device; CPU is the one that goes inactive and is never pre-compiled
    void switch_between_devices(bool compile_for_all, std::shared_ptr<ov::ICompiledModel>* result = nullptr) {
        if (compile_for_all) {
            config.insert(ov::intel_auto::compile_for_all(true));
        }
        expect_selected_devices({"GPU.0", "CPU", "GPU.0", "CPU"});
        std::shared_ptr<ov::ICompiledModel> compiled_model;
        OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
        run_inferences(compiled_model, 3);
        if (result) {
            *result = std::move(compiled_model);
        }
    }

    std::atomic<int> m_gpu0_compile_count{0};
    std::atomic<int> m_cpu_compile_count{0};
    std::atomic<int> m_cpu_cache_key_compute_count{0};
    std::filesystem::path m_cache_dir;
    bool m_write_cpu_cache_blob = true;
    bool m_write_gpu0_cache_blob = false;
    std::weak_ptr<ov::MockICompiledModel> m_cpu_model;
    std::weak_ptr<ov::mock_auto_plugin::MockISyncInferRequest> m_cpu_worker_request;
    std::function<std::vector<ov::ProfilingInfo>()> m_cpu_profiling_info;

    // returns the request that ran on CPU, after CPU was released by a request that ran on GPU.0
    std::shared_ptr<ov::IAsyncInferRequest> infer_then_release_cpu() {
        const std::vector<ov::PropertyName> supported_properties = {ov::compilation_num_threads,
                                                                    ov::enable_profiling};
        ON_CALL(*core, get_property(_, StrEq(ov::supported_properties.name()), _))
            .WillByDefault(Return(ov::Any(supported_properties)));
        config.insert(ov::enable_profiling(true));
        config.insert(ov::intel_auto::compile_for_all(true));
        expect_selected_devices({"GPU.0", "CPU", "GPU.0"});
        auto compiled_model = plugin->compile_model(model, config);
        // the first request ran on CPU and is kept alive without running again
        auto cpu_request = compiled_model->create_infer_request();
        cpu_request->infer();
        EXPECT_FALSE(m_cpu_model.expired());
        // a second request switches to GPU.0, which releases CPU
        auto gpu_request = compiled_model->create_infer_request();
        gpu_request->infer();
        EXPECT_TRUE(m_cpu_model.expired());
        EXPECT_TRUE(m_cpu_worker_request.expired());
        return cpu_request;
    }
};

TEST_F(AutoDynamicInactiveDeviceReleaseTest, inactive_device_is_compiled_again_when_selected_again) {
    switch_between_devices(true);
    EXPECT_EQ(m_cpu_compile_count.load(), 2);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, cpu_is_released_without_a_non_cpu_precompile_candidate) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,CPU"));
    switch_between_devices(true);
    EXPECT_EQ(m_cpu_compile_count.load(), 2);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, cpu_is_kept_when_cache_blob_was_not_written) {
    m_write_cpu_cache_blob = false;
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    switch_between_devices(true, &compiled_model);
    EXPECT_EQ(m_cpu_compile_count.load(), 1);
    EXPECT_FALSE(m_cpu_model.expired());
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, cpu_cache_key_is_computed_once_across_releases) {
    config.erase(ov::device::priorities.name());
    config.insert(ov::device::priorities("GPU.0,CPU"));
    config.insert(ov::intel_auto::compile_for_all(true));
    expect_selected_devices({"GPU.0", "CPU", "GPU.0", "CPU", "GPU.0"});

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 4);

    EXPECT_EQ(m_cpu_compile_count.load(), 2);
    // Two CPU compiles write the mock cache entry; the release probe computes the key only once.
    EXPECT_EQ(m_cpu_cache_key_compute_count.load(), 3);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, profiling_does_not_keep_the_inactive_device_alive) {
    m_cpu_profiling_info = [] {
        ov::ProfilingInfo info;
        info.status = ov::ProfilingInfo::Status::EXECUTED;
        info.real_time = std::chrono::microseconds(7);
        info.cpu_time = std::chrono::microseconds(5);
        info.node_name = "cpu_node";
        info.exec_type = "cpu_exec";
        info.node_type = "cpu_type";
        return std::vector<ov::ProfilingInfo>{info};
    };
    auto cpu_request = infer_then_release_cpu();
    std::vector<ov::ProfilingInfo> profiling;
    OV_ASSERT_NO_THROW(profiling = cpu_request->get_profiling_info());
    ASSERT_EQ(profiling.size(), 1u);
    EXPECT_EQ(profiling[0].status, ov::ProfilingInfo::Status::EXECUTED);
    EXPECT_EQ(profiling[0].real_time, std::chrono::microseconds(7));
    EXPECT_EQ(profiling[0].cpu_time, std::chrono::microseconds(5));
    EXPECT_EQ(profiling[0].node_name, "cpu_node");
    EXPECT_EQ(profiling[0].exec_type, "cpu_exec");
    EXPECT_EQ(profiling[0].node_type, "cpu_type");
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, profiling_error_is_deferred_to_the_profiling_query) {
    m_cpu_profiling_info = []() -> std::vector<ov::ProfilingInfo> {
        OPENVINO_THROW("backend profiling failure");
    };
    // infer() succeeds even though the backend fails to build the profiling report
    auto cpu_request = infer_then_release_cpu();
    try {
        cpu_request->get_profiling_info();
        FAIL() << "the captured profiling error must be rethrown";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("backend profiling failure"), std::string::npos);
    }
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, profiling_of_a_failed_inference_does_not_return_the_previous_one) {
    m_cpu_profiling_info = [] {
        ov::ProfilingInfo info;
        info.node_name = "cpu_node";
        return std::vector<ov::ProfilingInfo>{info};
    };
    auto request = infer_then_release_cpu();
    EXPECT_EQ(request->get_profiling_info().size(), 1u);

    // remote tensors of different devices make the request fail before it reaches any device
    const auto input = request->get_inputs().at(0);
    const auto output = request->get_outputs().at(0);
    auto remote_input =
        std::make_shared<DeviceTaggedRemoteTensor>("GPU.0", input.get_element_type(), input.get_shape());
    auto remote_output =
        std::make_shared<DeviceTaggedRemoteTensor>("CPU", output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(request->set_tensor(input, ov::SoPtr<ov::ITensor>{remote_input, nullptr}));
    OV_ASSERT_NO_THROW(request->set_tensor(output, ov::SoPtr<ov::ITensor>{remote_output, nullptr}));
    OV_ASSERT_NO_THROW(request->start_async());
    EXPECT_THROW(request->wait(), ov::Exception);

    EXPECT_THROW(request->get_profiling_info(), ov::Exception);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, device_without_cache_is_kept) {
    // The mock Core handles nested device properties, not the DEVICE_PROPERTIES_CPU shorthand.
    config.insert(ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::cache_dir("")}}}));
    switch_between_devices(true);
    EXPECT_EQ(m_cpu_compile_count.load(), 1);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, device_without_caching_support_is_kept) {
    ON_CALL(*core, device_supports_model_caching(StrEq(ov::test::utils::DEVICE_CPU))).WillByDefault(Return(false));
    switch_between_devices(true);
    EXPECT_EQ(m_cpu_compile_count.load(), 1);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, previous_device_is_released_before_the_next_device_is_compiled) {
    // GPU.1 is not pre-compiled, so it is only compiled by the switch
    config.insert(ov::device::properties(ov::AnyMap{{"GPU.1", ov::AnyMap{ov::cache_dir("")}}}));
    auto gpu1_compile_count = std::make_shared<std::atomic<int>>(0);
    auto cpu_alive_at_gpu1_compile = std::make_shared<std::atomic<int>>(0);
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault([this, gpu1_compile_count, cpu_alive_at_gpu1_compile](const std::shared_ptr<const ov::Model>&,
                                                                             const std::string&,
                                                                             const ov::AnyMap&) {
            ++*gpu1_compile_count;
            if (!m_cpu_model.expired()) {
                ++*cpu_alive_at_gpu1_compile;
            }
            return mockExeNetworkActual;
        });
    config.insert(ov::intel_auto::compile_for_all(true));
    expect_selected_devices({"GPU.0", "CPU", "GPU.1"});
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 2);
    EXPECT_EQ(gpu1_compile_count->load(), 1);
    // CPU was released before GPU.1 was compiled
    EXPECT_EQ(cpu_alive_at_gpu1_compile->load(), 0);
    EXPECT_TRUE(m_cpu_model.expired());
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, previous_device_is_reloaded_when_the_new_device_fails_to_compile) {
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                          _))
        .WillByDefault([](const std::shared_ptr<const ov::Model>&, const std::string&, const ov::AnyMap&)
                           -> ov::SoPtr<ov::ICompiledModel> {
            OPENVINO_THROW("mock GPU.1 compile failure");
        });
    config.insert(ov::intel_auto::compile_for_all(true));
    // GPU.1 fails on the switch away from CPU, and the fallback selects CPU again
    expect_selected_devices({"GPU.0", "CPU", "GPU.1", "CPU"});
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    run_inferences(compiled_model, 2);
    // CPU was released before GPU.1 was compiled, so the fallback loads it again
    EXPECT_EQ(m_cpu_compile_count.load(), 2);
    EXPECT_FALSE(m_cpu_model.expired());
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, initial_device_is_released_like_any_other_device) {
    m_write_gpu0_cache_blob = true;
    switch_between_devices(true);
    // released when the requests moved to CPU, and compiled again when GPU.0 was selected again
    EXPECT_EQ(m_gpu0_compile_count.load(), 2);
    EXPECT_EQ(m_cpu_compile_count.load(), 2);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, initial_device_is_kept_without_a_reloadable_cache) {
    switch_between_devices(true);
    EXPECT_EQ(m_gpu0_compile_count.load(), 1);
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, model_queries_follow_the_device_in_use_after_the_initial_device_release) {
    m_write_gpu0_cache_blob = true;
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    switch_between_devices(true, &compiled_model);
    // the requests run on CPU and GPU.0 has been released by now
    EXPECT_EQ(m_gpu0_compile_count.load(), 2);
    EXPECT_EQ(compiled_model->inputs().size(), model->inputs().size());
    EXPECT_EQ(compiled_model->outputs().size(), model->outputs().size());
    EXPECT_EQ(compiled_model->get_runtime_model(), model);
    EXPECT_EQ(compiled_model->get_property(ov::model_name.name()).as<std::string>(), "cpu_model");
    EXPECT_TRUE(compiled_model->get_property(ov::loaded_from_cache.name()).as<bool>());
    const auto device_properties = compiled_model->get_property(ov::device::properties.name()).as<ov::AnyMap>();
    ASSERT_EQ(device_properties.size(), 1u);
    EXPECT_EQ(device_properties.begin()->first, "CPU");
    EXPECT_EQ(device_properties.begin()->second.as<ov::AnyMap>().at(ov::model_name.name()).as<std::string>(),
              "cpu_model");
}

TEST_F(AutoDynamicInactiveDeviceReleaseTest, inactive_device_is_kept_without_compile_for_all) {
    switch_between_devices(false);
    EXPECT_EQ(m_cpu_compile_count.load(), 1);
    EXPECT_EQ(m_gpu0_compile_count.load(), 1);
}
}  // namespace ov::test
