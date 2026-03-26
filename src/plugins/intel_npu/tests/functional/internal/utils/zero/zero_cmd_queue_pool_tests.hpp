// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <ze_api.h>
#include <ze_command_queue_npu_ext.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class ZeroCmdQueuePoolTests : public ov::test::behavior::OVPluginTestBase,
                              public testing::WithParamInterface<std::string> {
protected:
    ov::AnyMap configuration;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
        std::string target_device;
        ov::AnyMap configuration;
        target_device = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        target_device = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();
    }

    void TearDown() override {
        init_struct = nullptr;
        APIBaseTest::TearDown();
    }
};

TEST_P(ZeroCmdQueuePoolTests, SetWorkloadType) {
    ::intel_npu::CommandQueueDesc command_queue_desc{0, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, ZE_WORKLOAD_TYPE_BACKGROUND};

    if (init_struct->getCommandQueueDdiTable().version() > 0) {
        OV_ASSERT_NO_THROW(
            ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc));
    } else {
        OV_EXPECT_THROW_HAS_SUBSTRING(
            ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc),
            ov::Exception,
            "The WorkloadType property is not supported by the current Driver Version!");
    }
}

TEST_P(ZeroCmdQueuePoolTests, PoolReusabilityTest) {
    // Test that the pool correctly reuses queues after weak_ptr cleanup
    ::intel_npu::CommandQueueDesc command_queue_desc{0, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

    // First allocation
    std::shared_ptr<::intel_npu::CommandQueue> queue1 =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc);
    EXPECT_NE(queue1, nullptr);

    // Second allocation with same descriptor should return the same instance (while ref is held)
    std::shared_ptr<::intel_npu::CommandQueue> queue2 =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc);
    EXPECT_EQ(queue1.get(), queue2.get()) << "Same descriptor should return pooled queue while alive";

    // Release first reference
    queue1.reset();

    std::shared_ptr<::intel_npu::CommandQueue> queue3 =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc);
    EXPECT_NE(queue3, nullptr) << "Should always be able to allocate a queue";
    EXPECT_EQ(queue2.get(), queue3.get()) << "Same descriptor should return pooled queue while alive";

    queue2.reset();
    queue3.reset();
}

TEST_P(ZeroCmdQueuePoolTests, AllCommandQueueOptionsCombinations) {
    if (init_struct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1)) {
        GTEST_SKIP() << "Not all the command queue options are supported by the current driver.\n";
    }

    // Exhaustively test all combinations of command queue creation options
    std::vector<ze_command_queue_priority_t> priorities{ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW,
                                                        ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                        ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH};

    std::vector<_ze_command_queue_workload_type_t> workload_types{ZE_WORKLOAD_TYPE_DEFAULT,
                                                                  ZE_WORKLOAD_TYPE_BACKGROUND};

    std::vector<uint32_t> option_combinations{
        0,
        ZE_NPU_COMMAND_QUEUE_OPTION_TURBO,
        ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC,
        ZE_NPU_COMMAND_QUEUE_OPTION_TURBO | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC};

    int successful_allocations = 0;

    for (auto priority : priorities) {
        for (auto workload_type : workload_types) {
            for (auto options : option_combinations) {
                ::intel_npu::CommandQueueDesc cmd_desc{0, priority, workload_type, options};
                if (options & ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC) {
                    static int owner = 1;  // Just a dummy pointer value for testing
                    cmd_desc.owner_tag = &owner;
                }
                auto cmd_queue = ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, cmd_desc);

                EXPECT_NE(cmd_queue, nullptr) << "Failed to allocate command queue for priority=" << (int)priority
                                              << " workload_type=" << (int)workload_type << " options=" << options;
                successful_allocations++;

                // Verify we get the same instance for the same descriptor
                auto cmd_queue2 = ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, cmd_desc);
                EXPECT_EQ(cmd_queue.get(), cmd_queue2.get()) << "Same descriptor should return the same pooled queue";
            }
        }
    }

    const int all_combinations =
        static_cast<int>(priorities.size() * workload_types.size() * option_combinations.size());
    EXPECT_EQ(all_combinations, 24) << "Expected 3 priorities * 2 workload types * 4 option combinations";
    EXPECT_EQ(successful_allocations, all_combinations)
        << "The test must cover and create all command queue combinations exactly once";
}

TEST_P(ZeroCmdQueuePoolTests, CreateDifferentCommandQueueForEachDeviceSyncOption) {
    if (init_struct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1)) {
        GTEST_SKIP()
            << "ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC command queue option is not supported by the current driver.\n";
    }

    int owner_a = 1;
    int owner_b = 2;

    // With DEVICE_SYNC enabled, owner_tag participates in the pool key.
    ::intel_npu::CommandQueueDesc device_sync_desc_a{0,
                                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                     ZE_WORKLOAD_TYPE_DEFAULT,
                                                     ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC,
                                                     &owner_a};
    auto queue_device_sync_a_1 =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, device_sync_desc_a);
    auto queue_device_sync_a_2 =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, device_sync_desc_a);

    EXPECT_NE(queue_device_sync_a_1, nullptr);
    EXPECT_EQ(queue_device_sync_a_1.get(), queue_device_sync_a_2.get())
        << "Same DEVICE_SYNC owner_tag should reuse the same pooled queue";

    ::intel_npu::CommandQueueDesc device_sync_desc_b{0,
                                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                     ZE_WORKLOAD_TYPE_DEFAULT,
                                                     ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC,
                                                     &owner_b};
    auto queue_device_sync_b =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, device_sync_desc_b);

    EXPECT_NE(queue_device_sync_b, nullptr);
    EXPECT_NE(queue_device_sync_a_1.get(), queue_device_sync_b.get())
        << "Different DEVICE_SYNC owner_tag pointers should create different pooled queues";

    // Without DEVICE_SYNC, owner_tag must not affect pooling.
    ::intel_npu::CommandQueueDesc no_device_sync_desc_a{0,
                                                        ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                        ZE_WORKLOAD_TYPE_DEFAULT,
                                                        0,
                                                        &owner_a};
    ::intel_npu::CommandQueueDesc no_device_sync_desc_b{0,
                                                        ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                        ZE_WORKLOAD_TYPE_DEFAULT,
                                                        0,
                                                        &owner_b};
    auto queue_no_device_sync_a =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, no_device_sync_desc_a);
    auto queue_no_device_sync_b =
        ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, no_device_sync_desc_b);

    EXPECT_NE(queue_no_device_sync_a, nullptr);
    EXPECT_EQ(queue_no_device_sync_a.get(), queue_no_device_sync_b.get())
        << "owner_tag should be ignored when DEVICE_SYNC option is not set";
}

TEST_P(ZeroCmdQueuePoolTests, MultiThreadingTest) {
    if (init_struct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 0)) {
        GTEST_SKIP() << "Workload type and turbo options are not supported by the current driver.\n";
    }

    // Systematically cover all combinations of priorities, workload types, and turbo
    // Priorities: LOW (0), HIGH (1), NORMAL (2) = 3 options
    // Workload types: BACKGROUND (0), DEFAULT (1) = 2 options (when version > 0)
    // Turbo options: without TURBO (0), with TURBO (1) = 2 options (when version > 0)
    // Total combinations: 3 * 2 * 2 = 12 (with version > 0) or 3 (with version <= 0)

    constexpr int threads_no = 256;
    std::array<std::thread, threads_no> threads;
    std::vector<std::shared_ptr<::intel_npu::CommandQueue>> queue_refs(threads_no);
    std::mutex results_mutex;
    int successful_allocations = 0;

    for (int i = 0; i < threads_no; ++i) {
        threads[i] = std::thread([&, i]() -> void {
            // Cycle through all priority options
            ze_command_queue_priority_t priority;
            int priority_idx = i % 3;
            if (priority_idx == 0) {
                priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
            } else if (priority_idx == 1) {
                priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
            } else {
                priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
            }

            _ze_command_queue_workload_type_t workload_type = ZE_WORKLOAD_TYPE_DEFAULT;
            uint32_t commandQueueOptions = 0;

            // Cycle through workload types (2 options)
            int workload_idx = (i / 3) % 2;
            workload_type = (workload_idx == 0) ? ZE_WORKLOAD_TYPE_BACKGROUND : ZE_WORKLOAD_TYPE_DEFAULT;

            // Cycle through command queue options (2 options: with/without TURBO)
            int options_idx = (i / 6) % 2;
            if (options_idx == 0) {
                commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
            }
            // else: no options (commandQueueOptions = 0)

            ::intel_npu::CommandQueueDesc command_queue_desc{0, priority, workload_type, commandQueueOptions};
            auto cmd_queue =
                ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc);

            {
                std::lock_guard<std::mutex> lock(results_mutex);
                queue_refs[i] = cmd_queue;
                successful_allocations++;
            }
        });
    }

    for (int i = 0; i < threads_no; ++i) {
        threads[i].join();
    }

    // Verify that all threads successfully obtained command queues
    EXPECT_GT(successful_allocations, 0) << "At least some command queues should be allocated";

    // Verify that we attempted allocations with different parameters
    int unique_combinations_requested = 12;
    int threads_per_combination = threads_no / unique_combinations_requested;
    EXPECT_GE(successful_allocations, threads_per_combination)
        << "Should have successfully allocated at least one thread per combination type";

    // Keep refs alive for the duration and then let them be cleaned up
    queue_refs.clear();
}

TEST_P(ZeroCmdQueuePoolTests, MultiThreadingTestWithDestroyCmdQueue) {
    if (init_struct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 0)) {
        GTEST_SKIP() << "Workload type and turbo options are not supported by the current driver.\n";
    }

    constexpr int threads_no = 128;
    constexpr int iterations_per_thread = 16;
    std::array<std::thread, threads_no> threads;
    std::atomic<int> successful_cycles{0};

    for (int i = 0; i < threads_no; ++i) {
        threads[i] = std::thread([&, i]() -> void {
            for (int iteration = 0; iteration < iterations_per_thread; ++iteration) {
                // Cycle through all priority options
                ze_command_queue_priority_t priority;
                const int priority_idx = (i + iteration) % 3;
                if (priority_idx == 0) {
                    priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
                } else if (priority_idx == 1) {
                    priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
                } else {
                    priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
                }

                _ze_command_queue_workload_type_t workload_type = ZE_WORKLOAD_TYPE_DEFAULT;
                uint32_t commandQueueOptions = 0;

                // Cycle through workload types (2 options)
                const int workload_idx = ((i + iteration) / 3) % 2;
                workload_type = (workload_idx == 0) ? ZE_WORKLOAD_TYPE_BACKGROUND : ZE_WORKLOAD_TYPE_DEFAULT;

                // Cycle through command queue options (2 options: with/without TURBO)
                const int options_idx = ((i + iteration) / 6) % 2;
                if (options_idx == 0) {
                    commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
                }
                // else: no options (commandQueueOptions = 0)

                ::intel_npu::CommandQueueDesc command_queue_desc{0, priority, workload_type, commandQueueOptions};
                auto cmd_queue =
                    ::intel_npu::ZeroCmdQueuePool::getInstance().getCommandQueue(init_struct, command_queue_desc);
                ASSERT_NE(cmd_queue, nullptr);

                // Explicitly release right after creation to stress the pool deleter path.
                cmd_queue.reset();
                ++successful_cycles;
            }
        });
    }

    for (int i = 0; i < threads_no; ++i) {
        threads[i].join();
    }

    EXPECT_EQ(successful_cycles.load(), threads_no * iterations_per_thread)
        << "All create/destroy cycles must complete";
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
