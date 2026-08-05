// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_options_cache.hpp"

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <optional>
#include <string>
#include <vector>

namespace {

TEST(CompilerOptionsCacheTests, UnknownOptionsAreReportedAsUnsupportedByDefault) {
    EXPECT_FALSE(intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                    "CACHE_TEST_UNKNOWN_OPTION",
                                                                    std::nullopt,
                                                                    std::nullopt));
    EXPECT_FALSE(intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                                    "CACHE_TEST_UNKNOWN_OPTION",
                                                                    std::nullopt,
                                                                    std::nullopt));
}

TEST(CompilerOptionsCacheTests, UnknownOptionValueCombinationIsUnsupportedByDefault) {
    EXPECT_FALSE(intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                    "CACHE_TEST_OPTION",
                                                                    std::optional<std::string>{"VALUE_A"},
                                                                    std::nullopt));
    EXPECT_FALSE(intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                                    "CACHE_TEST_OPTION",
                                                                    std::optional<std::string>{"VALUE_B"},
                                                                    std::nullopt));
}

TEST(CompilerOptionsCacheTests, RejectsNonExplicitCompilerType) {
    EXPECT_THROW(intel_npu::CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType::PREFER_PLUGIN,
                                                                    "CACHE_TEST_OPT",
                                                                    std::nullopt,
                                                                    std::nullopt),
                 ov::Exception);
}

}  // namespace
