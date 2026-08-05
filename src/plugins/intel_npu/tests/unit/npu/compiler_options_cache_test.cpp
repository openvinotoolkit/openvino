// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_options_cache.hpp"

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <optional>
#include <string>

namespace {

TEST(CompilerOptionsCacheTests, RejectsNonExplicitCompilerType) {
    try {
        (void)intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::PREFER_PLUGIN,
                                                                 "CACHE_TEST_OPT",
                                                                 std::optional<std::string>{},
                                                                 ov::SoPtr<intel_npu::IEngineBackend>{});
        FAIL() << "Expected exception for non-explicit compiler type";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("Expected DRIVER or PLUGIN"), std::string::npos);
    }
}

TEST(CompilerOptionsCacheTests, DriverCompilerRequiresBackend) {
    EXPECT_ANY_THROW((void)intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                              "CACHE_TEST_OPT",
                                                                              std::optional<std::string>{},
                                                                              ov::SoPtr<intel_npu::IEngineBackend>{}));
}

}  // namespace
