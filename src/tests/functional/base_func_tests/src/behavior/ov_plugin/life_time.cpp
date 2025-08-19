// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "behavior/ov_plugin/life_time.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"

namespace ov {
namespace test {
namespace behavior {
std::string OVHoldersTest::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device;
}

void OVHoldersTest::SetUp() {
    target_device = this->GetParam();
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
    if (deathTestStyle == "fast") {
        ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    }
    function = ov::test::utils::make_split_concat();
}

void OVHoldersTest::TearDown() {
    ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
    APIBaseTest::TearDown();
}

#define EXPECT_NO_CRASH(_statement) \
EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

static void release_order_test(std::vector<std::size_t> order, const std::string &deviceName,
                               std::shared_ptr<ov::Model> function) {
    ov::AnyVector objects;
    {
        ov::Core core = ov::test::utils::create_core();
        auto compiled_model = core.compile_model(function, deviceName);
        auto request = compiled_model.create_infer_request();

        objects = {core, compiled_model, request};
    }
    for (auto&& i : order) {
        objects.at(i) = {};
    }
}

#ifndef __EMSCRIPTEN__

TEST_P(OVHoldersTest, Orders) {
    std::vector<std::string> objects{ "core", "compiled_model", "request"};
    std::vector<std::size_t> order(objects.size());
    std::iota(order.begin(), order.end(), 0);
    do {
        std::stringstream order_str;
        for (auto&& i : order) {
            order_str << objects.at(i) << " ";
        }
        EXPECT_NO_CRASH(release_order_test(order, target_device, function)) << "for order: " << order_str.str();
    } while (std::next_permutation(order.begin(), order.end()));
}

#endif // __EMSCRIPTEN__

TEST_P(OVHoldersTest, LoadedState) {
    std::vector<ov::VariableState> states;
    {
        ov::Core core = ov::test::utils::create_core();
        auto compiled_model = core.compile_model(function, target_device);
        auto request = compiled_model.create_infer_request();
        try {
            states = request.query_state();
        } catch(...) {}
    }
}

TEST_P(OVHoldersTest, LoadedTensor) {
    ov::Tensor tensor;
    {
        ov::Core core = ov::test::utils::create_core();
        auto compiled_model = core.compile_model(function, target_device);
        auto request = compiled_model.create_infer_request();
        tensor = request.get_input_tensor();
    }
}

TEST_P(OVHoldersTest, LoadedAny) {
    ov::Any any;
    {
        ov::Core core = ov::test::utils::create_core();
        auto compiled_model = core.compile_model(function, target_device);
        any = compiled_model.get_property(ov::supported_properties.name());
    }
}

TEST_P(OVHoldersTest, LoadedRemoteContext) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::RemoteContext ctx;
    {
        ov::Core core = ov::test::utils::create_core();
        auto compiled_model = core.compile_model(function, target_device);
        try {
            ctx = compiled_model.get_context();
        } catch(...) {}
    }
}

TEST_P(OVHoldersTestWithConfig, LoadedTensor) {
    ov::Tensor tensor;
    {
        ov::Core core = ov::test::utils::create_core();
        ov::AnyMap property;
        property[ov::intel_auto::device_bind_buffer.name()] = true;
        if (target_device.find("AUTO") != std::string::npos)
            property[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
        auto compiled_model = core.compile_model(function, target_device, property);
        auto request = compiled_model.create_infer_request();
        tensor = request.get_input_tensor();
    }
}

std::string OVHoldersTestOnImportedNetwork::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device;
}

void OVHoldersTestOnImportedNetwork::SetUp() {
    target_device = this->GetParam();
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
    if (deathTestStyle == "fast") {
        ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    }
    function = ov::test::utils::make_split_concat();
}

void OVHoldersTestOnImportedNetwork::TearDown() {
    ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
    APIBaseTest::TearDown();
}

TEST_P(OVHoldersTestOnImportedNetwork, LoadedTensor) {
    ov::Core core = ov::test::utils::create_core();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, target_device);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, target_device);
    auto request = compiled_model.create_infer_request();
    ov::Tensor tensor = request.get_input_tensor();
}

TEST_P(OVHoldersTestOnImportedNetwork, CreateRequestWithCoreRemoved) {
    ov::Core core = ov::test::utils::create_core();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, target_device);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, target_device);
    core = ov::Core{};
    auto request = compiled_model.create_infer_request();
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
