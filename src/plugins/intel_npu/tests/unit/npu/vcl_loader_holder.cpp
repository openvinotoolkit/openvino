// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

namespace {

using namespace ::intel_npu;

const std::string kMissingLibraryDir = "npu_vcl_loader_holder_missing_dir";

TEST(VCLLoaderHolderTests, Construction_WithMissingLibrary_DoesNotLoad) {
    EXPECT_NO_THROW(VCLLoaderHolder{kMissingLibraryDir});
    EXPECT_NO_THROW(VCLLoaderHolder{});
}

TEST(VCLLoaderHolderTests, Functions_WithMissingLibrary_Throws) {
    const VCLLoaderHolder holder(kMissingLibraryDir);
    EXPECT_THROW(holder.functions(), ov::Exception);
}

TEST(VCLLoaderHolderTests, Functions_WithEmptyLibraryDir_Throws) {
    const VCLLoaderHolder holder{std::string{}};
    EXPECT_THROW(holder.functions(), ov::Exception);
}

class VCLLoaderHolderFailedLoadTests : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    ("npu_vcl_loader_holder_" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(directory);
    }

    void TearDown() override {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

    std::filesystem::path directory;
};

TEST_F(VCLLoaderHolderFailedLoadTests, Functions_AfterFailedLoad_RethrowsFirstErrorWithoutLoadingAgain) {
    const VCLLoaderHolder holder(ov::util::path_to_string(directory));
    std::string firstError;
    try {
        holder.functions();
        FAIL() << "The library is missing, the first call must throw";
    } catch (const ov::Exception& ex) {
        firstError = ex.what();
    }

    // A new load attempt would now fail on the invalid file with a different error than on the missing one.
    const auto library = ov::util::make_plugin_library_name(directory, "openvino_intel_npu_compiler_loader");
    std::ofstream(library) << "not a shared library";

    try {
        holder.functions();
        FAIL() << "A failed load must be remembered, the second call must throw";
    } catch (const ov::Exception& ex) {
        EXPECT_EQ(firstError, ex.what());
    }
}

TEST(VCLLoaderHolderTests, Functions_CalledConcurrentlyWithMissingLibrary_EveryCallThrows) {
    const VCLLoaderHolder holder(kMissingLibraryDir);
    constexpr size_t threadCount = 8;
    std::atomic<size_t> throwCount{0};
    std::promise<void> startSignal;
    const auto start = startSignal.get_future().share();

    std::vector<std::thread> threads;
    for (size_t i = 0; i < threadCount; i++) {
        threads.emplace_back([&] {
            start.wait();
            try {
                holder.functions();
            } catch (const ov::Exception&) {
                throwCount++;
            }
        });
    }
    startSignal.set_value();
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(throwCount.load(), threadCount);
}

class VCLLoaderHolderLibraryTests : public ::testing::Test {
protected:
    void SetUp() override {
        directory = ov::test::utils::getOpenvinoLibDirectory();
        const auto library =
            ov::util::make_plugin_library_name(std::filesystem::path(directory), "openvino_intel_npu_compiler_loader");
        if (!std::filesystem::exists(library)) {
            GTEST_SKIP() << "Compiler library not installed: " << library;
        }
    }

    std::string directory;
};

TEST_F(VCLLoaderHolderLibraryTests, Functions_AfterHolderDestroyed_LibraryRemainsCallableUntilTableReleased) {
    std::shared_ptr<const VCLFunctionTable> functions;
    {
        const VCLLoaderHolder holder(directory);
        functions = holder.functions();
    }
    ASSERT_NE(functions, nullptr);
    std::weak_ptr<const VCLFunctionTable> owner = functions;
    vcl_version_info_t compilerVersion{};
    vcl_version_info_t profilingVersion{};
    ASSERT_NE(functions->vclGetVersion, nullptr);
    EXPECT_EQ(functions->vclGetVersion(&compilerVersion, &profilingVersion), VCL_RESULT_SUCCESS);
    functions.reset();
    EXPECT_TRUE(owner.expired());
}

TEST_F(VCLLoaderHolderLibraryTests, Functions_ConcurrentFirstCalls_ReturnSameTable) {
    const VCLLoaderHolder holder(directory);
    constexpr size_t threadCount = 8;
    std::promise<void> startSignal;
    const auto start = startSignal.get_future().share();
    std::array<std::future<std::shared_ptr<const VCLFunctionTable>>, threadCount> calls;
    for (auto& call : calls) {
        call = std::async(std::launch::async, [&holder, start] {
            start.wait();
            return holder.functions();
        });
    }
    startSignal.set_value();
    const auto functions = calls.front().get();
    ASSERT_NE(functions, nullptr);
    for (size_t i = 1; i < calls.size(); i++) {
        EXPECT_EQ(functions, calls[i].get());
    }
}

}  // namespace
