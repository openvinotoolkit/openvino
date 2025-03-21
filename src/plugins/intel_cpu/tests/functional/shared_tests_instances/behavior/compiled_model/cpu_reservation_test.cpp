// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <thread>

#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "openvino/openvino.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"

#if defined(_WIN32)
#    include <windows.h>
#endif

using namespace testing;
using Device = std::string;
using Config = ov::AnyMap;
using CpuReservationTest = ::testing::Test;
// Issue: 163348
using DISABLED_CpuReservationTest = ::testing::Test;

TEST_F(DISABLED_CpuReservationTest, Mutiple_CompiledModel_Reservation) {
    std::vector<std::shared_ptr<ov::Model>> models;
    Config config = {ov::enable_profiling(true)};
    Device target_device(ov::test::utils::DEVICE_CPU);
    std::atomic<unsigned int> counter{0u};

    models.emplace_back(ov::test::utils::make_2_input_subtract());
    models.emplace_back(ov::test::utils::make_multi_single_conv());

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    core->set_property(target_device, config);
    ov::AnyMap property_config_reserve = {{ov::num_streams.name(), ov::streams::Num(1)},
                                          {ov::inference_num_threads.name(), 1},
                                          {ov::hint::enable_cpu_reservation.name(), true}};
    ov::AnyMap property_config = {{ov::num_streams.name(), ov::streams::Num(1)}, {ov::inference_num_threads.name(), 1}};

    std::vector<std::thread> threads(2);
    for (auto& thread : threads) {
        thread = std::thread([&]() {
            auto value = counter++;
            auto compiled_model = core->compile_model(models[value % models.size()],
                                                      target_device,
                                                      value == 1 ? property_config : property_config_reserve);
            auto cpu_reservation = compiled_model.get_property(ov::hint::enable_cpu_reservation.name());
            auto num_streams = compiled_model.get_property(ov::num_streams.name());
            ASSERT_EQ(cpu_reservation, value == 1 ? false : true);
            ASSERT_EQ(num_streams, ov::streams::Num(1));
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}

TEST_F(DISABLED_CpuReservationTest, Cpu_Reservation_NoAvailableCores) {
    std::vector<std::shared_ptr<ov::Model>> models;
    Config config = {ov::enable_profiling(true)};
    Device target_device(ov::test::utils::DEVICE_CPU);
    models.emplace_back(ov::test::utils::make_2_input_subtract());

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    core->set_property(target_device, config);
    ov::AnyMap property_config = {{ov::num_streams.name(), 1},
                                  {ov::inference_num_threads.name(), 2000},
                                  {ov::hint::enable_hyper_threading.name(), true},
                                  {ov::hint::enable_cpu_reservation.name(), true}};
    auto compiled_model = core->compile_model(models[0], target_device, property_config);
    EXPECT_THROW(core->compile_model(models[0], target_device, property_config), ov::Exception);
}

#if defined(__linux__)
TEST_F(DISABLED_CpuReservationTest, Cpu_Reservation_CpuPinning) {
    std::vector<std::shared_ptr<ov::Model>> models;
    Config config = {ov::enable_profiling(true)};
    Device target_device(ov::test::utils::DEVICE_CPU);
    models.emplace_back(ov::test::utils::make_2_input_subtract());
    bool cpu_pinning = false;

#if defined(__linux__)
    cpu_pinning = true;
#elif defined(_WIN32)
    ULONG highestNodeNumber = 0;
    if (!GetNumaHighestNodeNumber(&highestNodeNumber)) {
        std::cout << "Error getting highest NUMA node number: " << GetLastError() << std::endl;
        return;
    }
    if (highestNodeNumber > 0) {
        cpu_pinning = false;
    } else {
        cpu_pinning = true;
    }
#endif

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    core->set_property(target_device, config);
    ov::AnyMap property_config = {{ov::inference_num_threads.name(), 1},
                                  {ov::hint::enable_cpu_reservation.name(), true}};
    auto compiled_model = core->compile_model(models[0], target_device, property_config);
    auto res_cpu_pinning = compiled_model.get_property(ov::hint::enable_cpu_pinning.name());
    ASSERT_EQ(res_cpu_pinning, cpu_pinning);
}

TEST_F(CpuReservationTest, Cpu_Reservation_CompiledModel_Release) {
    std::vector<std::shared_ptr<ov::Model>> models;
    Config config = {ov::enable_profiling(true)};
    Device target_device(ov::test::utils::DEVICE_CPU);
    models.emplace_back(ov::test::utils::make_2_input_subtract());

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    core->set_property(target_device, config);
    ov::AnyMap property_config = {{ov::num_streams.name(), 2000},
                                  {ov::inference_num_threads.name(), 2000},
                                  {ov::hint::enable_hyper_threading.name(), true},
                                  {ov::hint::enable_cpu_reservation.name(), true}};
    {
        auto compiled_model = core->compile_model(models[0], target_device, property_config);
        EXPECT_THROW(core->compile_model(models[0], target_device, property_config), ov::Exception);
    }

    ov::AnyMap reserve_1_config = {{ov::num_streams.name(), 1},
                                  {ov::inference_num_threads.name(), 1},
                                  {ov::hint::enable_cpu_reservation.name(), true}};
    EXPECT_NO_THROW(core->compile_model(models[0], target_device, reserve_1_config));
}

#endif
