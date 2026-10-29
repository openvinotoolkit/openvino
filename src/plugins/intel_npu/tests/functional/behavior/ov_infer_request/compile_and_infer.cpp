// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <common_test_utils/test_assertions.hpp>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "common/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> getConstantGraph(element::Type type) {
    ResultVector results;
    ParameterVector params;
    auto op = std::make_shared<ov::op::v1::Add>(opset8::Constant::create(type, {1}, {1}),
                                                opset8::Constant::create(type, {1}, {1}));
    op->set_friendly_name("Add");
    auto res = std::make_shared<ov::op::v0::Result>(op);
    res->set_friendly_name("Result");
    res->get_output_tensor(0).set_names({"tensor_output"});
    results.push_back(res);
    return std::make_shared<Model>(results, params);
}

inline std::shared_ptr<ov::Model> createSimpleModel() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto add_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(data, add_constant);

    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{data});
}

inline std::shared_ptr<ov::Model> createModelContainingSubgraph() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    input->get_output_tensor(0).set_names({"tensor_input"});

    // Simple subgraph that adds 1 to the input
    std::shared_ptr<ov::Model> body = createSimpleModel();
    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_sliced_input(
        std::dynamic_pointer_cast<ov::op::v0::Parameter>(body->input().get_node_shared_ptr()),
        input,
        0,
        1,
        1,
        -1,
        0);

    // Outside the subgraph, add 1 to the output
    auto add_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(tensor_iterator->get_iter_value(body->output(), -1), add_constant);

    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->get_output_tensor(0).set_names({"tensor_output"});

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

inline bool isCommandQueueExtSupported() {
    return std::make_shared<::intel_npu::ZeroInitStructsHolder>()->getCommandQueueDdiTable().version() > 0;
}

inline bool isCommandQueueQosSupported() {
    return ::intel_npu::ZeroInitStructsHolder::getInstance()->isCommandQueueQosSupported();
}

// All the levels which can be explicitly requested by the application; QoS::DEFAULT is excluded since it reverts to
// the driver/OS-selected level.
const std::vector<ov::QoS> explicitQosLevels = {ov::QoS::HIGH,
                                                ov::QoS::MEDIUM,
                                                ov::QoS::LOW,
                                                ov::QoS::UTILITY,
                                                ov::QoS::ECO};

typedef std::tuple<std::shared_ptr<ov::Model>,  // Model
                   std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndInferRequestParams;

class OVCompileAndInferRequest : public testing::WithParamInterface<CompileAndInferRequestParams>,
                                 public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompileAndInferRequestParams>& obj) {
        std::shared_ptr<ov::Model> model;
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(model, targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }
    void SetUp() override {
        std::tie(function, target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabled_test_patterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
    }

protected:
    ov::CompiledModel execNet;
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
};

TEST_P(OVCompileAndInferRequest, AsyncInferRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(exception_ptr, nullptr);
        is_called = true;
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_TRUE(is_called);
}

TEST_P(OVCompileAndInferRequest, EvictGraphMemoryThenInferAgain) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());

    OV_ASSERT_NO_THROW(execNet.release_memory());

    OV_ASSERT_NO_THROW(req.infer());
}

// Disabled test that checks that memory eviction works correctly and does not cause memory leaks after inference runs.
// The test is disabled due to the fact that it requires another check of the memory since device_alloc_mem_size doesn't
// provide memory feedback after eviction is called.
TEST_P(OVCompileAndInferRequest, DISABLED_CheckMemoryAfterEvictGraphMemoryAndAfterRunningInferAgain) {
    if (std::make_shared<::intel_npu::ZeroInitStructsHolder>()->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 16)) {
        GTEST_SKIP() << "Memory eviction is not supported by the current driver version.";
    }

    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());

    uint64_t device_alloc_mem_size = 0;
    OV_ASSERT_NO_THROW(
        device_alloc_mem_size =
            core->get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()).as<uint64_t>());

    OV_ASSERT_NO_THROW(execNet.release_memory());

    uint64_t device_alloc_mem_size_after_evict = 0;
    OV_ASSERT_NO_THROW(
        device_alloc_mem_size_after_evict =
            core->get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()).as<uint64_t>());

    OV_ASSERT_NO_THROW(req.infer());

    uint64_t device_alloc_mem_size_final = 0;
    OV_ASSERT_NO_THROW(
        device_alloc_mem_size_final =
            core->get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()).as<uint64_t>());

    ASSERT_EQ(device_alloc_mem_size, device_alloc_mem_size_final);
    ASSERT_LT(device_alloc_mem_size_after_evict, device_alloc_mem_size);
}

TEST_P(OVCompileAndInferRequest, PluginWorkloadType) {
    configuration[workload_type.name()] = WorkloadType::DEFAULT;
    auto supportedProperties = core->get_property("NPU", supported_properties.name()).as<std::vector<PropertyName>>();
    bool workloadTypeSupported =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == workload_type.name();
        });

    if (isCommandQueueExtSupported()) {
        ASSERT_TRUE(workloadTypeSupported);
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

        const auto properties = execNet.get_property(supported_properties.name()).as<std::vector<PropertyName>>();
        ASSERT_TRUE(std::any_of(properties.begin(), properties.end(), [](const PropertyName& property) {
            return property == workload_type.name();
        }));

        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        ASSERT_FALSE(workloadTypeSupported);
        OV_EXPECT_THROW_HAS_SUBSTRING(
            core->compile_model(function, target_device, configuration),
            ov::Exception,
            "[ NOT_FOUND ] Option 'WORKLOAD_TYPE' is not supported for current configuration");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadType) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::AnyMap modelConfiguration;
    modelConfiguration[workload_type.name()] = WorkloadType::DEFAULT;
    auto supportedProperties = execNet.get_property(supported_properties.name()).as<std::vector<PropertyName>>();
    bool workloadTypeSupported =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == workload_type.name();
        });

    if (isCommandQueueExtSupported()) {
        ASSERT_TRUE(workloadTypeSupported);
        OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        ASSERT_FALSE(workloadTypeSupported);
        OV_EXPECT_THROW_HAS_SUBSTRING(execNet.set_property(modelConfiguration),
                                      ov::Exception,
                                      "Unsupported configuration key: WORKLOAD_TYPE");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadTypeDelayedExecutor) {
    configuration[intel_npu::defer_weights_load.name()] = true;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::AnyMap modelConfiguration;
    modelConfiguration[workload_type.name()] = WorkloadType::DEFAULT;

    if (isCommandQueueExtSupported()) {
        OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        OV_EXPECT_THROW_HAS_SUBSTRING(execNet.set_property(modelConfiguration),
                                      ov::Exception,
                                      "Unsupported configuration key: WORKLOAD_TYPE");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelAndCreateMultipleInferRequestsWithDelayedExecutor) {
    configuration[intel_npu::defer_weights_load.name()] = true;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::InferRequest req0, req1, req2;
    OV_ASSERT_NO_THROW(req0 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req1 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = execNet.create_infer_request());

    OV_ASSERT_NO_THROW(req0.infer());
    OV_ASSERT_NO_THROW(req1.infer());
    OV_ASSERT_NO_THROW(req2.infer());
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadTypeUpdateAfterCompilation) {
    if (!isCommandQueueExtSupported()) {
        GTEST_SKIP() << "Workload type update is not supported with current driver.\n";
    }

    configuration[workload_type.name()] = WorkloadType::DEFAULT;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    ASSERT_EQ(execNet.get_property(workload_type.name()).as<WorkloadType>(), WorkloadType::DEFAULT);
    ov::AnyMap modelConfiguration;
    modelConfiguration[workload_type.name()] = WorkloadType::EFFICIENT;
    OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
    ASSERT_EQ(execNet.get_property(workload_type.name()).as<WorkloadType>(), WorkloadType::EFFICIENT);
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(exception_ptr, nullptr);
        is_called = true;
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_TRUE(is_called);
}

TEST_P(OVCompileAndInferRequest, CompiledModelModelPriorityUpdateAfterCompilation) {
    configuration[hint::model_priority.name()] = hint::Priority::MEDIUM;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ASSERT_EQ(execNet.get_property(hint::model_priority.name()).as<hint::Priority>(), hint::Priority::MEDIUM);

    ov::AnyMap modelConfiguration;
    ov::InferRequest req;

    modelConfiguration[hint::model_priority.name()] = hint::Priority::HIGH;
    OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
    ASSERT_EQ(execNet.get_property(hint::model_priority.name()).as<hint::Priority>(), hint::Priority::HIGH);
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());

    modelConfiguration[hint::model_priority.name()] = hint::Priority::LOW;
    OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
    ASSERT_EQ(execNet.get_property(hint::model_priority.name()).as<hint::Priority>(), hint::Priority::LOW);
    OV_ASSERT_NO_THROW(req.infer());
}

using OVCompileAndInferRequestMultiThreading = OVCompileAndInferRequest;

TEST_P(OVCompileAndInferRequestMultiThreading, CreateInferRequestsAndRunOnDifferentThreads) {
    configuration[intel_npu::defer_weights_load.name()] = true;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    const int num_threads = 64;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions;
    std::mutex exceptions_mutex;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &exceptions, &exceptions_mutex]() {
            try {
                auto req = execNet.create_infer_request();
                req.infer();
            } catch (...) {
                std::lock_guard<std::mutex> lock(exceptions_mutex);
                exceptions.emplace_back(std::current_exception());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (!exceptions.empty()) {
        try {
            std::rethrow_exception(exceptions.front());
        } catch (const std::exception& ex) {
            FAIL() << ex.what();
        } catch (...) {
            FAIL() << "Unknown exception occurred in one of the threads.";
        }
    }
}

TEST_P(OVCompileAndInferRequestMultiThreading, CreateInferRequestsAndRunOnDifferentThreadsWithWorkloadTypeAndPriority) {
    if (!isCommandQueueExtSupported()) {
        GTEST_SKIP() << "Workload type update is not supported with current driver.\n";
    }

    configuration[intel_npu::defer_weights_load.name()] = true;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    int num_threads = 32;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions;
    std::mutex exceptions_mutex;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &exceptions, &exceptions_mutex, i]() {
            try {
                // Stagger threads to desynchronize property setting and expose race conditions
                std::this_thread::sleep_for(std::chrono::milliseconds(i % 10));
                if (i % 2 == 0) {
                    execNet.set_property(ov::workload_type(WorkloadType::DEFAULT));
                } else {
                    execNet.set_property(ov::workload_type(WorkloadType::EFFICIENT));
                }
                if (i % 3 == 0) {
                    execNet.set_property(ov::hint::model_priority(ov::hint::Priority::HIGH));
                } else if (i % 3 == 1) {
                    execNet.set_property(ov::hint::model_priority(ov::hint::Priority::MEDIUM));
                } else {
                    execNet.set_property(ov::hint::model_priority(ov::hint::Priority::LOW));
                }
                auto req = execNet.create_infer_request();
                req.infer();
            } catch (...) {
                std::lock_guard<std::mutex> lock(exceptions_mutex);
                exceptions.emplace_back(std::current_exception());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (!exceptions.empty()) {
        try {
            std::rethrow_exception(exceptions.front());
        } catch (const std::exception& ex) {
            FAIL() << ex.what();
        } catch (...) {
            FAIL() << "Unknown exception occurred in one of the threads.";
        }
    }
}

TEST_P(OVCompileAndInferRequestMultiThreading,
       CompileModelCreateInferRequestsAndRunOnDifferentThreadsWithWorkloadTypeAndPriority) {
    if (!isCommandQueueExtSupported()) {
        GTEST_SKIP() << "Workload type update is not supported with current driver.\n";
    }

    configuration[intel_npu::defer_weights_load.name()] = true;

    int num_threads = 32;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions;
    std::vector<ov::CompiledModel> compiled_models(num_threads);
    std::mutex exceptions_mutex;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &exceptions, &exceptions_mutex, &compiled_models, i]() {
            try {
                compiled_models[i] = core->compile_model(function, target_device, configuration);
                // Stagger threads to desynchronize property setting and expose race conditions
                std::this_thread::sleep_for(std::chrono::milliseconds(i % 10));
                if (i % 2 == 0) {
                    compiled_models[i].set_property(ov::workload_type(WorkloadType::DEFAULT));
                } else {
                    compiled_models[i].set_property(ov::workload_type(WorkloadType::EFFICIENT));
                }
                if (i % 3 == 0) {
                    compiled_models[i].set_property(ov::hint::model_priority(ov::hint::Priority::HIGH));
                } else if (i % 3 == 1) {
                    compiled_models[i].set_property(ov::hint::model_priority(ov::hint::Priority::MEDIUM));
                } else {
                    compiled_models[i].set_property(ov::hint::model_priority(ov::hint::Priority::LOW));
                }
                auto req = compiled_models[i].create_infer_request();
                req.infer();
            } catch (...) {
                std::lock_guard<std::mutex> lock(exceptions_mutex);
                exceptions.emplace_back(std::current_exception());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (!exceptions.empty()) {
        try {
            std::rethrow_exception(exceptions.front());
        } catch (const std::exception& ex) {
            FAIL() << ex.what();
        } catch (...) {
            FAIL() << "Unknown exception occurred in one of the threads.";
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        if (i % 2 == 0) {
            ASSERT_EQ(compiled_models[i].get_property(ov::workload_type.name()).as<WorkloadType>(),
                      WorkloadType::DEFAULT);
        } else {
            ASSERT_EQ(compiled_models[i].get_property(ov::workload_type.name()).as<WorkloadType>(),
                      WorkloadType::EFFICIENT);
        }

        if (i % 3 == 0) {
            ASSERT_EQ(compiled_models[i].get_property(ov::hint::model_priority.name()).as<ov::hint::Priority>(),
                      ov::hint::Priority::HIGH);
        } else if (i % 3 == 1) {
            ASSERT_EQ(compiled_models[i].get_property(ov::hint::model_priority.name()).as<ov::hint::Priority>(),
                      ov::hint::Priority::MEDIUM);
        } else {
            ASSERT_EQ(compiled_models[i].get_property(ov::hint::model_priority.name()).as<ov::hint::Priority>(),
                      ov::hint::Priority::LOW);
        }
    }
}

TEST_P(OVCompileAndInferRequestMultiThreading,
       CompileModelCreateInferRequestsAndRunOnDifferentThreadsWithWorkloadTypePriorityAndSharedCommonQueueDisabled) {
    if (!isCommandQueueExtSupported()) {
        GTEST_SKIP() << "Workload type update is not supported with current driver.\n";
    }

    configuration[intel_npu::defer_weights_load.name()] = true;
    configuration[intel_npu::shared_common_queue.name()] = false;

    int num_threads = 32;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions;
    std::vector<ov::CompiledModel> compiled_models(num_threads);
    std::mutex exceptions_mutex;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &exceptions, &exceptions_mutex, &compiled_models, i]() {
            try {
                compiled_models[i] = core->compile_model(function, target_device, configuration);
                // Stagger threads to desynchronize property setting and expose race conditions
                std::this_thread::sleep_for(std::chrono::milliseconds(i % 10));
                if (i % 2 == 0) {
                    compiled_models[i].set_property(ov::workload_type(WorkloadType::DEFAULT));
                } else {
                    compiled_models[i].set_property(ov::workload_type(WorkloadType::EFFICIENT));
                }
                if (i % 3 == 0) {
                    compiled_models[i].set_property(ov::hint::model_priority(ov::hint::Priority::HIGH));
                } else if (i % 3 == 1) {
                    compiled_models[i].set_property(ov::hint::model_priority(ov::hint::Priority::MEDIUM));
                } else {
                    compiled_models[i].set_property(ov::hint::model_priority(ov::hint::Priority::LOW));
                }
                auto req = compiled_models[i].create_infer_request();
                req.infer();
            } catch (...) {
                std::lock_guard<std::mutex> lock(exceptions_mutex);
                exceptions.emplace_back(std::current_exception());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (!exceptions.empty()) {
        try {
            std::rethrow_exception(exceptions.front());
        } catch (const std::exception& ex) {
            FAIL() << ex.what();
        } catch (...) {
            FAIL() << "Unknown exception occurred in one of the threads.";
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        if (i % 2 == 0) {
            ASSERT_EQ(compiled_models[i].get_property(ov::workload_type.name()).as<WorkloadType>(),
                      WorkloadType::DEFAULT);
        } else {
            ASSERT_EQ(compiled_models[i].get_property(ov::workload_type.name()).as<WorkloadType>(),
                      WorkloadType::EFFICIENT);
        }

        if (i % 3 == 0) {
            ASSERT_EQ(compiled_models[i].get_property(ov::hint::model_priority.name()).as<ov::hint::Priority>(),
                      ov::hint::Priority::HIGH);
        } else if (i % 3 == 1) {
            ASSERT_EQ(compiled_models[i].get_property(ov::hint::model_priority.name()).as<ov::hint::Priority>(),
                      ov::hint::Priority::MEDIUM);
        } else {
            ASSERT_EQ(compiled_models[i].get_property(ov::hint::model_priority.name()).as<ov::hint::Priority>(),
                      ov::hint::Priority::LOW);
        }
    }
}

using OVCompileAndInferRequestTurbo = OVCompileAndInferRequest;

TEST_P(OVCompileAndInferRequestTurbo, CompiledModelTurbo) {
    configuration[intel_npu::turbo.name()] = true;

    auto supportedProperties = core->get_property("NPU", supported_properties.name()).as<std::vector<PropertyName>>();
    bool isTurboSupported =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == intel_npu::turbo.name();
        });

    if (isCommandQueueExtSupported()) {
        ASSERT_TRUE(isTurboSupported);
        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
        auto turbosetting_compiled_model = execNet.get_property(intel_npu::turbo.name());
        OV_ASSERT_NO_THROW(turbosetting_compiled_model = true);
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        OV_EXPECT_THROW_HAS_SUBSTRING(core->compile_model(function, target_device, configuration),
                                      ov::Exception,
                                      "[ NOT_FOUND ] Option 'NPU_TURBO' is not supported for current configuration");
    }
}

class OVCompileAndInferRequestQoS : public OVCompileAndInferRequest {
protected:
    void SetUp() override {
        OVCompileAndInferRequest::SetUp();
        if (!isCommandQueueQosSupported()) {
            GTEST_SKIP() << "The QoS feature is not supported by the current driver.\n";
        }
    }

    static ov::QoSInfo getQosInfo(const ov::CompiledModel& compiledModel) {
        ov::QoSInfo qosInfo{};
        try {
            qosInfo = compiledModel.get_property(ov::qos_info);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Getting " << ov::qos_info.name() << " failed: " << e.what();
        }
        return qosInfo;
    }

    static void expectRequestedQos(const ov::CompiledModel& compiledModel, ov::QoS expectedLevel) {
        const auto qosInfo = getQosInfo(compiledModel);
        EXPECT_EQ(qosInfo.level, expectedLevel);
        EXPECT_TRUE(qosInfo.is_overridden) << "A level requested by the application must be reported as overridden";
    }

    static void expectDriverDefaultQos(const ov::CompiledModel& compiledModel) {
        const auto qosInfo = getQosInfo(compiledModel);
        EXPECT_NE(qosInfo.level, ov::QoS::DEFAULT) << "The effective level is never DEFAULT";
        EXPECT_FALSE(qosInfo.is_overridden) << "No level was requested, so it must not be reported as overridden";
    }

    static void inferAsync(ov::InferRequest& req) {
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    }
};

TEST_P(OVCompileAndInferRequestQoS, QosPropertiesAreReportedAsSupported) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    auto supportedProperties = execNet.get_property(supported_properties);
    auto qosIt = std::find(supportedProperties.begin(), supportedProperties.end(), ov::qos.name());
    ASSERT_NE(qosIt, supportedProperties.end());
    EXPECT_TRUE(qosIt->is_mutable()) << "QOS must be writable";
    auto qosInfoIt = std::find(supportedProperties.begin(), supportedProperties.end(), ov::qos_info.name());
    ASSERT_NE(qosInfoIt, supportedProperties.end());
    EXPECT_FALSE(qosInfoIt->is_mutable()) << "QOS_INFO must be read-only";

    auto pluginProperties = core->get_property(target_device, supported_properties);
    EXPECT_NE(std::find(pluginProperties.begin(), pluginProperties.end(), ov::qos.name()), pluginProperties.end());
}

TEST_P(OVCompileAndInferRequestQoS, QosIsWriteOnly) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    ov::Any value;
    OV_ASSERT_NO_THROW(value = execNet.get_property(ov::qos.name()));
    EXPECT_TRUE(value.empty());
    OV_ASSERT_NO_THROW(value = core->get_property(target_device, ov::qos.name()));
    EXPECT_TRUE(value.empty());
}

TEST_P(OVCompileAndInferRequestQoS, QosInfoIsReadOnly) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    OV_EXPECT_THROW(execNet.set_property({{ov::qos_info.name(), ov::QoSInfo{ov::QoS::LOW, true}}}),
                    ov::Exception,
                    testing::_);
}

TEST_P(OVCompileAndInferRequestQoS, QosInfoWithoutRequestedQosReportsDriverDefault) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    expectDriverDefaultQos(execNet);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    inferAsync(req);

    expectDriverDefaultQos(execNet);
}

TEST_P(OVCompileAndInferRequestQoS, QosAtCompileTimeIsReportedAndUsedForInference) {
    for (auto level : explicitQosLevels) {
        auto compileConfiguration = configuration;
        compileConfiguration[ov::qos.name()] = level;

        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, compileConfiguration));
        // Queried before any infer request is created, the graph may not be initialized yet.
        expectRequestedQos(execNet, level);

        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        inferAsync(req);

        expectRequestedQos(execNet, level);
    }
}

TEST_P(OVCompileAndInferRequestQoS, DefaultQosAtCompileTimeReportsDriverDefault) {
    auto compileConfiguration = configuration;
    compileConfiguration[ov::qos.name()] = ov::QoS::DEFAULT;

    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, compileConfiguration));
    expectDriverDefaultQos(execNet);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    inferAsync(req);

    expectDriverDefaultQos(execNet);
}

TEST_P(OVCompileAndInferRequestQoS, QosUpdateAfterCompilation) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    inferAsync(req);

    for (auto level : explicitQosLevels) {
        OV_ASSERT_NO_THROW(execNet.set_property({ov::qos(level)}));
        expectRequestedQos(execNet, level);

        // Both an already existing request and a newly created one must keep working with the updated level.
        inferAsync(req);
        ov::InferRequest newReq;
        OV_ASSERT_NO_THROW(newReq = execNet.create_infer_request());
        inferAsync(newReq);

        expectRequestedQos(execNet, level);
    }

    // Setting DEFAULT reverts to the driver/OS-selected level.
    OV_ASSERT_NO_THROW(execNet.set_property({ov::qos(ov::QoS::DEFAULT)}));
    expectDriverDefaultQos(execNet);
    inferAsync(req);
    expectDriverDefaultQos(execNet);
}

TEST_P(OVCompileAndInferRequestQoS, QosUpdateBeforeFirstInferRequest) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    // No infer request yet: with deferred weights load the graph is not initialized, so the level must still be
    // applied and reported.
    OV_ASSERT_NO_THROW(execNet.set_property({ov::qos(ov::QoS::LOW)}));
    expectRequestedQos(execNet, ov::QoS::LOW);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    inferAsync(req);

    expectRequestedQos(execNet, ov::QoS::LOW);
}

TEST_P(OVCompileAndInferRequestQoS, QosOverridesTheCompileTimeLevel) {
    auto compileConfiguration = configuration;
    compileConfiguration[ov::qos.name()] = ov::QoS::HIGH;

    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, compileConfiguration));
    expectRequestedQos(execNet, ov::QoS::HIGH);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    inferAsync(req);

    OV_ASSERT_NO_THROW(execNet.set_property({ov::qos(ov::QoS::ECO)}));
    expectRequestedQos(execNet, ov::QoS::ECO);
    inferAsync(req);
}

TEST_P(OVCompileAndInferRequestQoS, QosOfOneCompiledModelDoesNotAffectAnother) {
    ov::CompiledModel otherExecNet;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    OV_ASSERT_NO_THROW(otherExecNet = core->compile_model(function, target_device, configuration));

    ov::InferRequest req;
    ov::InferRequest otherReq;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(otherReq = otherExecNet.create_infer_request());
    inferAsync(req);
    inferAsync(otherReq);

    OV_ASSERT_NO_THROW(execNet.set_property({ov::qos(ov::QoS::UTILITY)}));
    expectRequestedQos(execNet, ov::QoS::UTILITY);
    expectDriverDefaultQos(otherExecNet);

    inferAsync(req);
    inferAsync(otherReq);
}

TEST_P(OVCompileAndInferRequestQoS, QosSetOnCoreIsUsedByCompiledModels) {
    // Use a dedicated core so the plugin-level property does not leak into the other tests.
    ov::Core localCore;
    OV_ASSERT_NO_THROW(localCore.set_property(target_device, ov::qos(ov::QoS::MEDIUM)));

    OV_ASSERT_NO_THROW(execNet = localCore.compile_model(function, target_device, configuration));
    expectRequestedQos(execNet, ov::QoS::MEDIUM);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    inferAsync(req);
    expectRequestedQos(execNet, ov::QoS::MEDIUM);

    // A compile-time configuration takes precedence over the plugin-level one.
    auto compileConfiguration = configuration;
    compileConfiguration[ov::qos.name()] = ov::QoS::ECO;
    OV_ASSERT_NO_THROW(execNet = localCore.compile_model(function, target_device, compileConfiguration));
    expectRequestedQos(execNet, ov::QoS::ECO);
}

TEST_P(OVCompileAndInferRequestQoS, QosAcceptsStringValues) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    OV_ASSERT_NO_THROW(execNet.set_property({{ov::qos.name(), "UTILITY"}}));
    expectRequestedQos(execNet, ov::QoS::UTILITY);

    OV_EXPECT_THROW(execNet.set_property({{ov::qos.name(), "NOT_A_QOS_LEVEL"}}), ov::Exception, testing::_);
}

TEST_P(OVCompileAndInferRequestQoS, QosChangesWhileInferringOnOtherThreads) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

    constexpr int numThreads = 8;
    constexpr int numIterations = 10;
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&]() {
            try {
                auto req = execNet.create_infer_request();
                for (int iteration = 0; iteration < numIterations; ++iteration) {
                    req.infer();
                }
            } catch (...) {
                failed = true;
            }
        });
    }

    for (int iteration = 0; iteration < numIterations; ++iteration) {
        const auto level = explicitQosLevels[iteration % explicitQosLevels.size()];
        EXPECT_NO_THROW(execNet.set_property({ov::qos(level)}));
        EXPECT_NO_THROW(execNet.get_property(ov::qos_info));
    }

    for (auto& thread : threads) {
        thread.join();
    }
    ASSERT_FALSE(failed) << "Inference must not fail while QoS is changed on another thread";

    OV_ASSERT_NO_THROW(execNet.set_property({ov::qos(ov::QoS::LOW)}));
    expectRequestedQos(execNet, ov::QoS::LOW);
}

using OVCompileAndInferRequestQoSNotSupported = OVCompileAndInferRequest;

TEST_P(OVCompileAndInferRequestQoSNotSupported, QosIsRejectedWhenDriverDoesNotSupportIt) {
    if (isCommandQueueQosSupported()) {
        GTEST_SKIP() << "The QoS feature is supported by the current driver.\n";
    }

    auto supportedProperties = core->get_property(target_device, supported_properties);
    EXPECT_EQ(std::find(supportedProperties.begin(), supportedProperties.end(), ov::qos.name()),
              supportedProperties.end());

    auto compileConfiguration = configuration;
    compileConfiguration[ov::qos.name()] = ov::QoS::LOW;
    OV_EXPECT_THROW_HAS_SUBSTRING(core->compile_model(function, target_device, compileConfiguration),
                                  ov::Exception,
                                  "is not supported");

    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    OV_EXPECT_THROW(execNet.set_property({ov::qos(ov::QoS::LOW)}), ov::Exception, testing::_);
    OV_EXPECT_THROW(execNet.get_property(ov::qos_info), ov::Exception, testing::_);
}

using OVCompileAndInferRequestSerializers = OVCompileAndInferRequest;

TEST_P(OVCompileAndInferRequestSerializers, AccurateResults) {
    try {
        execNet = core->compile_model(function, target_device, configuration);
        ov::InferRequest inference_request;
        OV_ASSERT_NO_THROW(inference_request = execNet.create_infer_request());

        const ov::Tensor input = utils::create_tensor(ov::element::f32, ov::Shape{1}, std::vector<float>{1.0f});
        inference_request.set_input_tensor(input);
        OV_ASSERT_NO_THROW(inference_request.infer());

        const ov::Tensor output = inference_request.get_tensor("tensor_output");
        const ov::Tensor expected = utils::create_tensor(ov::element::f32, ov::Shape{1}, std::vector<float>{3.0f});
        OV_ASSERT_NO_THROW(utils::compare(expected, output));
    } catch (const ov::Exception& exception) {
        ASSERT_STR_CONTAINS(
            exception.what(),
            "[ NOT_FOUND ] Option 'NPU_MODEL_SERIALIZER_VERSION' is not supported for current configuration");
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequest,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequest>);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVCompileAndInferRequestMultiThreading,
    ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(ov::test::utils::mergeConfigs(configs,
                                                                         ov::test::utils::quietCompilerLogsConfig))),
    ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestMultiThreading>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequestTurbo,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::defer_weights_load(true)},
                                                {ov::intel_npu::defer_weights_load(false)}})),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestTurbo>);

// Cover both the shared common queue (queue taken from the pool on each submission) and the per compiled model queue,
// each with the graph initialized at compile time and with the initialization deferred to the first infer request.
const std::vector<ov::AnyMap> qosConfigs = {
    {ov::intel_npu::shared_common_queue(true), ov::intel_npu::defer_weights_load(false)},
    {ov::intel_npu::shared_common_queue(true), ov::intel_npu::defer_weights_load(true)},
    {ov::intel_npu::shared_common_queue(false), ov::intel_npu::defer_weights_load(false)},
    {ov::intel_npu::shared_common_queue(false), ov::intel_npu::defer_weights_load(true)}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequestQoS,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(qosConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestQoS>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequestQoSNotSupported,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestQoSNotSupported>);

INSTANTIATE_TEST_SUITE_P(
    compatibility_smoke_BehaviorTests,
    OVCompileAndInferRequestSerializers,
    ::testing::Combine(
        ::testing::Values(createModelContainingSubgraph()),
        ::testing::Values(ov::test::utils::DEVICE_NPU),
        ::testing::ValuesIn(std::vector<ov::AnyMap>{
            {ov::intel_npu::model_serializer_version(ov::intel_npu::ModelSerializerVersion::AUTO)},
            {ov::intel_npu::model_serializer_version(ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY)},
            {ov::intel_npu::model_serializer_version(ov::intel_npu::ModelSerializerVersion::NO_WEIGHTS_COPY)}})),
    ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestSerializers>);

}  // namespace
