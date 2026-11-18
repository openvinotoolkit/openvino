// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <future>
#include <memory>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/util/file_util.hpp"

namespace {

using namespace ::intel_npu;

class VCLLoaderTests : public ::testing::Test {
protected:
    void SetUp() override {
        const auto library =
            ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), "openvino_intel_npu_compiler_loader");
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

}  // namespace
