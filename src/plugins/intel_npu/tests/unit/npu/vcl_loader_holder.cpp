// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"

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

TEST(VCLLoaderHolderTests, Functions_AfterFailedLoad_ThrowsAgainInsteadOfCachingTheFailure) {
    const ::intel_npu::VCLLoaderHolder holder(kMissingLibraryDir);
    EXPECT_THROW(holder.functions(), ov::Exception);
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

}  // namespace
