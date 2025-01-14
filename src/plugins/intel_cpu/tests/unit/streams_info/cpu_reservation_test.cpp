
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <thread>
#include <string>

#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "openvino/runtime/properties.hpp"
// #include "openvino/util/file_util.hpp"
#include "openvino/openvino.hpp"

using Device = std::string;
using Config = ov::AnyMap;
using CpuReservationTest = ::testing::Test;

TEST_F(CpuReservationTest, Mutiple_CompiledModel_Reservation) {
    std::vector<std::shared_ptr<ov::Model>> models;
    Config config = {ov::enable_profiling(true)};
    Device target_device("CPU");
    std::atomic<unsigned int> counter{0u};

    models.emplace_back(ov::test::utils::make_2_input_subtract());
    models.emplace_back(ov::test::utils::make_multi_single_conv());

    ov::Core core;
    core.set_property(target_device, config);
    ov::AnyMap property_config_reserve = {{ov::num_streams.name(), 1},
                                          {ov::inference_num_threads.name(), 1},
                                          {ov::hint::enable_cpu_reservation.name(), true}};
    ov::AnyMap property_config = {{ov::num_streams.name(), 1}, {ov::inference_num_threads.name(), 1}};

    std::vector<std::thread> threads(2);
    for (auto& thread : threads) {
        thread = std::thread([&]() {
            auto value = counter++;
            (void)core.compile_model(models[value % models.size()],
                                     target_device,
                                     value == 1 ? property_config : property_config_reserve);
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}