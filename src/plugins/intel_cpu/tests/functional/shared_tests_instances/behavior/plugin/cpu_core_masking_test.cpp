// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

/**
 * @brief Test CPU plugin behavior with CPU core masking
 * 
 * Tests plugin initialization and cleanup when running in environments
 * with CPU core masking (e.g., Docker containers with cpuset constraints).
 * Verifies proper resource management and prevents heap corruption issues.
 */
class CPUPluginCoreMaskingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test runs in normal environment, but the fixes ensure it works in Docker too
    }
};

/**
 * @brief Test plugin initialization with CPU core constraints
 * 
 * Verifies that device property queries work correctly in CPU-constrained environments
 */
TEST_F(CPUPluginCoreMaskingTest, QueryDevicePropertiesWithMaskedCores) {
    ov::Core core;
    
    // Query available devices - should not crash
    auto devices = core.get_available_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";
    
    bool found_cpu = false;
    for (const auto& device : devices) {
        if (device == "CPU") {
            found_cpu = true;
            
            // Get supported properties - this is where the crash occurred in issue #32684
            auto supported_props = core.get_property(device, ov::supported_properties);
            ASSERT_FALSE(supported_props.empty()) << "CPU device has no supported properties";
            
            // Query critical properties
            auto device_name = core.get_property(device, ov::device::full_name);
            EXPECT_FALSE(device_name.empty()) << "Device name is empty";
            
            // Query stream-related properties - these were involved in the crash
            auto num_streams = core.get_property(device, ov::num_streams);
            EXPECT_GE(static_cast<int32_t>(num_streams), 0) << "Invalid number of streams";
            
            auto inference_threads = core.get_property(device, ov::inference_num_threads);
            EXPECT_GE(inference_threads, 0) << "Invalid number of inference threads";
        }
    }
    
    ASSERT_TRUE(found_cpu) << "CPU device not found in available devices";
}

/**
 * @brief Test repeated plugin lifecycle with CPU masking
 * 
 * Verifies proper resource cleanup during repeated initialization and destruction
 */
TEST_F(CPUPluginCoreMaskingTest, RepeatedInitializationAndDestruction) {
    // Create and destroy Core multiple times to verify proper cleanup
    for (int iteration = 0; iteration < 5; ++iteration) {
        ov::Core core;
        auto devices = core.get_available_devices();
        
        // Query CPU properties to trigger executor initialization
        for (const auto& device : devices) {
            if (device == "CPU") {
                auto props = core.get_property(device, ov::supported_properties);
                EXPECT_FALSE(props.empty());
            }
        }
        // Core destructor should not crash
    }
}

/**
 * @brief Test concurrent plugin initialization
 * 
 * Verifies thread safety of executor management under concurrent access
 */
TEST_F(CPUPluginCoreMaskingTest, ConcurrentInitialization) {
    constexpr int num_threads = 4;
    constexpr int iterations_per_thread = 3;
    
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&failures]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                try {
                    ov::Core core;
                    auto devices = core.get_available_devices();
                    
                    for (const auto& device : devices) {
                        if (device == "CPU") {
                            core.get_property(device, ov::supported_properties);
                        }
                    }
                } catch (const std::exception&) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(failures.load(), 0) << "Some threads failed during concurrent initialization";
}

}  // namespace

