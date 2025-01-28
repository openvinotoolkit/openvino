// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <thread>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

using Device = std::string;
using Config = ov::AnyMap;
using GpuReservationTest = ::testing::Test;

TEST_F(GpuReservationTest, Mutiple_CompiledModel_Reservation) {
    std::vector<std::shared_ptr<ov::Model>> models;
    Config config = {ov::enable_profiling(true)};
    std::vector<Device> target_devices = {ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU};
    std::atomic<unsigned int> counter{0u};

    models.emplace_back(ov::test::utils::make_2_input_subtract());
    models.emplace_back(ov::test::utils::make_multi_single_conv());

    auto core = ov::test::utils::PluginCache::get().core();
    core->set_property(target_devices[1], config);

    ov::AnyMap property_config = {{ov::num_streams.name(), 1},
                                  {ov::inference_num_threads.name(), 1},
                                  {ov::hint::enable_cpu_reservation.name(), true}};
    ov::AnyMap property_config_gpu = {{ov::num_streams.name(), ov::streams::Num(1)},
                                      {ov::hint::enable_cpu_reservation.name(), true}};

    std::vector<std::thread> threads(2);
    for (auto& thread : threads) {
        thread = std::thread([&]() {
            auto value = counter++;
            auto compiled_model = core->compile_model(models[value % models.size()],
                                                      target_devices[value % target_devices.size()],
                                                      value == 0 ? property_config : property_config_gpu);
            auto num_streams = compiled_model.get_property(ov::num_streams.name());
            auto cpu_reservation = compiled_model.get_property(ov::hint::enable_cpu_reservation.name());
            ASSERT_EQ(num_streams, ov::streams::Num(1));
            ASSERT_EQ(cpu_reservation, true);
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}