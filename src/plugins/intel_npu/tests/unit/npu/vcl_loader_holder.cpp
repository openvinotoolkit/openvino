// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

namespace {

const std::string kMissingLibraryDir = "npu_vcl_loader_holder_missing_dir";

TEST(VCLLoaderHolderTests, Construction_WithMissingLibrary_DoesNotLoad) {
    EXPECT_NO_THROW(::intel_npu::VCLLoaderHolder{kMissingLibraryDir});
    EXPECT_NO_THROW(::intel_npu::VCLLoaderHolder{});
}

TEST(VCLLoaderHolderTests, Functions_WithMissingLibrary_Throws) {
    const ::intel_npu::VCLLoaderHolder holder(kMissingLibraryDir);
    EXPECT_THROW(holder.functions(), ov::Exception);
}

TEST(VCLLoaderHolderTests, Functions_WithEmptyLibraryDir_Throws) {
    const ::intel_npu::VCLLoaderHolder holder{std::string{}};
    EXPECT_THROW(holder.functions(), ov::Exception);
}

TEST(VCLLoaderHolderTests, Functions_CalledConcurrentlyWithMissingLibrary_EveryCallThrows) {
    const ::intel_npu::VCLLoaderHolder holder(kMissingLibraryDir);
    constexpr size_t threadCount = 8;
    std::atomic<size_t> throwCount{0};

    std::vector<std::thread> threads;
    for (size_t i = 0; i < threadCount; i++) {
        threads.emplace_back([&] {
            try {
                holder.functions();
            } catch (const ov::Exception&) {
                throwCount++;
            }
        });
    }
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
    std::shared_ptr<const ::intel_npu::VCLFunctionTable> functions;
    {
        const ::intel_npu::VCLLoaderHolder holder(directory);
        functions = holder.functions();
    }
    ASSERT_NE(functions, nullptr);
    std::weak_ptr<const ::intel_npu::VCLFunctionTable> owner = functions;
    vcl_version_info_t compilerVersion{};
    vcl_version_info_t profilingVersion{};
    ASSERT_NE(functions->vclGetVersion, nullptr);
    EXPECT_EQ(functions->vclGetVersion(&compilerVersion, &profilingVersion), VCL_RESULT_SUCCESS);
    functions.reset();
    EXPECT_TRUE(owner.expired());
}

TEST_F(VCLLoaderHolderLibraryTests, Functions_ConcurrentFirstCalls_ReturnSameTable) {
    const ::intel_npu::VCLLoaderHolder holder(directory);
    constexpr size_t threadCount = 8;
    std::array<std::future<std::shared_ptr<const ::intel_npu::VCLFunctionTable>>, threadCount> calls;
    for (auto& call : calls) {
        call = std::async(std::launch::async, [&holder] {
            return holder.functions();
        });
    }
    const auto functions = calls.front().get();
    ASSERT_NE(functions, nullptr);
    for (size_t i = 1; i < calls.size(); i++) {
        EXPECT_EQ(functions, calls[i].get());
    }
}

}  // namespace
