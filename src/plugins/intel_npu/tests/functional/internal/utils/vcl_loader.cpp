// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/util/file_util.hpp"

namespace {

using namespace ::intel_npu;

constexpr std::string_view kLoaderLibrary = "openvino_intel_npu_compiler_loader";
constexpr std::string_view kLoadAttemptMarker = "Try to load";

class VCLLoaderTests : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        const auto library =
            ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), std::string(kLoaderLibrary));
        if (!std::filesystem::exists(library)) {
            GTEST_SKIP() << "Compiler library not installed: " << library;
        }
    }
};

TEST_F(VCLLoaderTests, FunctionTableKeepsLoaderAliveAfterOwnerReleased) {
    std::shared_ptr<const VCLFunctionTable> functions;
    {
        const auto loader = std::make_shared<VCLLoader>();
        functions = loader->sharedFunctions();
    }
    ASSERT_NE(functions, nullptr);
    ASSERT_NE(functions->vclGetVersion, nullptr);
    vcl_version_info_t compilerVersion{};
    vcl_version_info_t profilingVersion{};
    EXPECT_EQ(functions->vclGetVersion(&compilerVersion, &profilingVersion), VCL_RESULT_SUCCESS);
}

TEST_F(VCLLoaderTests, RepeatedCallsReturnTheSameTable) {
    const auto loader = std::make_shared<VCLLoader>();

    EXPECT_EQ(loader->sharedFunctions(), loader->sharedFunctions());
}

TEST_F(VCLLoaderTests, ConcurrentCallsReturnTheSameTable) {
    const auto loader = std::make_shared<VCLLoader>();
    constexpr std::size_t threadCount = 8;
    std::promise<void> startSignal;
    const auto start = startSignal.get_future().share();
    std::array<std::future<std::shared_ptr<const VCLFunctionTable>>, threadCount> calls;
    for (auto& call : calls) {
        call = std::async(std::launch::async, [&loader, start] {
            start.wait();
            return loader->sharedFunctions();
        });
    }
    startSignal.set_value();
    const auto functions = calls.front().get();
    ASSERT_NE(functions, nullptr);
    for (std::size_t i = 1; i < calls.size(); i++) {
        EXPECT_EQ(functions, calls[i].get());
    }
}

// Each plugin instance loads the VCL library on its own, at most once, and replays a failed load. The tests count the
// load attempts that the plugin logs, which holds whether or not the compiler library is installed.
class PluginVCLLoadingTests : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        logCallbackGuard.emplace(logCallback);
    }

    void TearDown() override {
        if (!cores.empty()) {
            cores.back()->set_property(ov::test::utils::DEVICE_NPU, ov::log::level(originalLogLevel));
        }
    }

    // The log level is global to the plugin library and a new plugin instance resets it, so it is raised after the
    // instance is created.
    ov::Core& createCore() {
        auto& core = *cores.emplace_back(std::make_unique<ov::Core>());
        const auto level = core.get_property(ov::test::utils::DEVICE_NPU, ov::log::level);
        if (cores.size() == 1) {
            originalLogLevel = level;
        }
        core.set_property(ov::test::utils::DEVICE_NPU, ov::log::level(ov::log::Level::DEBUG));
        return core;
    }

    std::size_t libraryLoadAttempts() {
        std::lock_guard<std::mutex> lock(messagesMutex);
        return static_cast<std::size_t>(std::count_if(messages.begin(), messages.end(), [](const std::string& message) {
            return message.find(kLoadAttemptMarker) != std::string::npos &&
                   message.find(kLoaderLibrary) != std::string::npos;
        }));
    }

    static std::string outcomeOf(const std::function<void()>& request) {
        try {
            request();
            return "ok";
        } catch (const std::exception& ex) {
            return ex.what();
        }
    }

    // Reaches the compiler-in-plugin through the property manager, which serializes its callers.
    static void requestCompilerVersion(ov::Core& core) {
        core.get_property(ov::test::utils::DEVICE_NPU,
                          ov::intel_npu::compiler_version.name(),
                          {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)});
    }

    // Reaches the compiler-in-plugin straight from the plugin, so concurrent requests are not serialized.
    static void requestQueryModel(ov::Core& core, const std::shared_ptr<const ov::Model>& model) {
        core.query_model(model,
                         ov::test::utils::DEVICE_NPU,
                         {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                          ov::intel_npu::platform(ov::intel_npu::Platform::NPU4000)});
    }

    std::vector<std::string> messages;
    std::mutex messagesMutex;
    std::function<void(std::string_view)> logCallback = [this](std::string_view message) {
        std::lock_guard<std::mutex> lock(messagesMutex);
        messages.emplace_back(message);
    };
    std::optional<ov::test::utils::LogCallbackGuard> logCallbackGuard;
    std::vector<std::unique_ptr<ov::Core>> cores;
    ov::log::Level originalLogLevel = ov::log::Level::NO;
};

TEST_F(PluginVCLLoadingTests, RepeatedRequestsLoadTheLibraryOnce) {
    auto& core = createCore();

    std::vector<std::string> outcomes;
    for (int i = 0; i < 3; i++) {
        outcomes.push_back(outcomeOf([&core] {
            requestCompilerVersion(core);
        }));
    }

    EXPECT_EQ(libraryLoadAttempts(), 1u);
    for (const auto& outcome : outcomes) {
        EXPECT_EQ(outcome, outcomes.front());
    }
}

TEST_F(PluginVCLLoadingTests, ConcurrentFirstRequestsLoadTheLibraryOnce) {
    constexpr std::size_t threadCount = 4;
    auto& core = createCore();
    const auto model = ov::test::utils::make_multi_single_conv();

    std::promise<void> startSignal;
    const auto start = startSignal.get_future().share();
    std::vector<std::future<std::string>> requests;
    for (std::size_t i = 0; i < threadCount; i++) {
        requests.push_back(std::async(std::launch::async, [&core, &model, start] {
            start.wait();
            return outcomeOf([&core, &model] {
                requestQueryModel(core, model);
            });
        }));
    }
    startSignal.set_value();
    for (auto& request : requests) {
        request.wait();
    }

    EXPECT_EQ(libraryLoadAttempts(), 1u);
}

TEST_F(PluginVCLLoadingTests, EachPluginInstanceLoadsTheLibraryItself) {
    auto& first = createCore();
    outcomeOf([&first] {
        requestCompilerVersion(first);
    });
    auto& second = createCore();
    outcomeOf([&second] {
        requestCompilerVersion(second);
    });

    EXPECT_EQ(libraryLoadAttempts(), 2u);
}

}  // namespace
