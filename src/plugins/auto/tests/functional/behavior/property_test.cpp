// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, default_perfmode_for_multi) {
    auto compiled_model =
        core.compile_model(model_cannot_batch, "MULTI", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")});
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode), ov::hint::PerformanceMode::THROUGHPUT);
}

TEST_F(AutoFuncTests, respect_secondary_property_for_multi) {
    auto compiled_model = core.compile_model(
        model_cannot_batch,
        "MULTI",
        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
         ov::device::properties("MOCK_GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties("MOCK_CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))});
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode), ov::hint::PerformanceMode::THROUGHPUT);
    auto prop = compiled_model.get_property(ov::device::properties.name()).as<ov::AnyMap>();
    for (auto& item : prop) {
        for (auto& item2 : item.second.as<ov::AnyMap>()) {
            if (item2.first == ov::hint::performance_mode) {
                if (item.first == "MOCK_CPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::LATENCY);
                } else if (item.first == "MOCK_GPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::THROUGHPUT);
                }
            }
        }
    }
}

TEST_F(AutoFuncTests, default_perfmode_for_auto_ctput) {
    auto compiled_model =
        core.compile_model(model_cannot_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode),
              ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
    auto prop = compiled_model.get_property(ov::device::properties.name()).as<ov::AnyMap>();
    for (auto& item : prop) {
        for (auto& item2 : item.second.as<ov::AnyMap>()) {
            if (item2.first == ov::hint::performance_mode) {
                if (item.first == "MOCK_CPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::THROUGHPUT);
                } else if (item.first == "MOCK_GPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::THROUGHPUT);
                }
            }
        }
    }
}

TEST_F(AutoFuncTests, default_perfmode_for_auto) {
    auto compiled_model =
        core.compile_model(model_cannot_batch, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")});
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode), ov::hint::PerformanceMode::LATENCY);
    auto prop = compiled_model.get_property(ov::device::properties.name()).as<ov::AnyMap>();
    for (auto& item : prop) {
        for (auto& item2 : item.second.as<ov::AnyMap>()) {
            if (item2.first == ov::hint::performance_mode) {
                if (item.first == "MOCK_CPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::LATENCY);
                } else if (item.first == "MOCK_GPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::LATENCY);
                }
            }
        }
    }
}

TEST_F(AutoFuncTests, respect_secondary_property_auto_ctput) {
    auto compiled_model = core.compile_model(
        model_cannot_batch,
        "AUTO",
        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
         ov::device::properties("MOCK_GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties("MOCK_CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))});
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode),
              ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
    auto prop = compiled_model.get_property(ov::device::properties.name()).as<ov::AnyMap>();
    for (auto& item : prop) {
        for (auto& item2 : item.second.as<ov::AnyMap>()) {
            if (item2.first == ov::hint::performance_mode) {
                if (item.first == "MOCK_CPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::LATENCY);
                } else if (item.first == "MOCK_GPU") {
                    EXPECT_EQ(item2.second, ov::hint::PerformanceMode::THROUGHPUT);
                }
            }
        }
    }
}