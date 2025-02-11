// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "behavior/ov_plugin/life_time.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {

class OVHoldersTestNPU : public OVPluginTestBase, public testing::WithParamInterface<CompilationParams> {
protected:
    ov::AnyMap configuration;
    std::string deathTestStyle;
    std::shared_ptr<ov::Model> function;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');

        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
               << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast") {
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ov::test::utils::make_conv_pool_relu();
    }

    void TearDown() override {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }
};

#define EXPECT_NO_CRASH(_statement) EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

static void release_order_test(std::vector<std::size_t> order, const std::string& deviceName,
                               std::shared_ptr<ov::Model> function, ov::AnyMap configuration) {
    ov::AnyVector objects;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, deviceName, configuration);
        auto request = compiled_model.create_infer_request();

        objects = {core, compiled_model, request};
    }
    for (auto&& i : order) {
        objects.at(i) = {};
    }
}

#ifndef __EMSCRIPTEN__

TEST_P(OVHoldersTestNPU, Orders) {
    std::vector<std::string> objects{"core", "compiled_model", "request"};
    std::vector<std::size_t> order(objects.size());
    std::iota(order.begin(), order.end(), 0);
    do {
        std::stringstream order_str;
        for (auto&& i : order) {
            order_str << objects.at(i) << " ";
        }
        EXPECT_NO_CRASH(release_order_test(order, target_device, function, configuration))
                << "for order: " << order_str.str();
    } while (std::next_permutation(order.begin(), order.end()));
}

#endif  // __EMSCRIPTEN__

TEST_P(OVHoldersTestNPU, LoadedState) {
    std::vector<ov::VariableState> states;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device, configuration);
        auto request = compiled_model.create_infer_request();
        try {
            states = request.query_state();
        } catch (...) {
        }
    }
}

TEST_P(OVHoldersTestNPU, LoadedInferRequest) {
    ov::InferRequest inferRequest;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device, configuration);
        inferRequest = compiled_model.create_infer_request();
    }
}

TEST_P(OVHoldersTestNPU, LoadedTensor) {
    ov::Tensor tensor;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device, configuration);
        auto request = compiled_model.create_infer_request();
        tensor = request.get_input_tensor();
    }
}

TEST_P(OVHoldersTestNPU, LoadedAny) {
    ov::Any any;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device, configuration);
        any = compiled_model.get_property(ov::supported_properties.name());
    }
}

TEST_P(OVHoldersTestNPU, LoadedRemoteContext) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::RemoteContext ctx;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device, configuration);
        try {
            ctx = compiled_model.get_context();
        } catch (...) {
        }
    }
}

class OVHoldersTestOnImportedNetworkNPU :
        public OVPluginTestBase,
        public testing::WithParamInterface<CompilationParams> {
protected:
    ov::AnyMap configuration;
    std::string deathTestStyle;
    std::shared_ptr<ov::Model> function;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');

        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
               << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast") {
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ov::test::utils::make_conv_pool_relu();
    }
    void TearDown() override {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }
};

TEST_P(OVHoldersTestOnImportedNetworkNPU, LoadedTensor) {
    ov::Core core = createCoreWithTemplate();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, target_device, configuration);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, target_device, configuration);
    auto request = compiled_model.create_infer_request();
    ov::Tensor tensor = request.get_input_tensor();
}

TEST_P(OVHoldersTestOnImportedNetworkNPU, CreateRequestWithCoreRemoved) {
    ov::Core core = createCoreWithTemplate();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, target_device, configuration);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, target_device, configuration);
    core = ov::Core{};
    auto request = compiled_model.create_infer_request();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
