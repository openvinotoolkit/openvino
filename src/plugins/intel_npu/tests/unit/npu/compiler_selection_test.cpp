// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_selection.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace {

using ::intel_npu::isPluginCompilerPreferred;

TEST(CompilerSelectionTests, WindowsMeteorLakeUsesDriverCompatibilityBoundary) {
    struct TestCase {
        std::optional<uint32_t> driverVersion;
        bool expected;
    };

    const std::array<TestCase, 6> testCases{{
        {0u, false},
        {2564u, false},
        {2565u, true},
        {2566u, true},
        {std::numeric_limits<uint32_t>::max(), true},
        {std::nullopt, false},
    }};

    for (const auto& testCase : testCases) {
        EXPECT_EQ(isPluginCompilerPreferred(ov::intel_npu::Platform::NPU3720,
                                            testCase.driverVersion,
                                            true),
                  testCase.expected);
    }
}

TEST(CompilerSelectionTests, LinuxMeteorLakeIgnoresDriverVersionEncoding) {
    const std::array<std::optional<uint32_t>, 4> driverVersions{{
        0u,
        2564u,
        std::numeric_limits<uint32_t>::max(),
        std::nullopt,
    }};

    for (const auto driverVersion : driverVersions) {
        EXPECT_TRUE(isPluginCompilerPreferred(ov::intel_npu::Platform::NPU3720, driverVersion, false));
    }
}

TEST(CompilerSelectionTests, KnownPlatformsPreferPluginOnBothOperatingSystems) {
    const std::array<std::string_view, 4> platforms{{
        ov::intel_npu::Platform::NPU4000,
        ov::intel_npu::Platform::NPU5010,
        ov::intel_npu::Platform::NPU5020,
        ov::intel_npu::Platform::NPU6010,
    }};
    const std::array<std::optional<uint32_t>, 3> driverVersions{{
        std::nullopt,
        0u,
        std::numeric_limits<uint32_t>::max(),
    }};

    for (const auto platform : platforms) {
        for (const auto driverVersion : driverVersions) {
            EXPECT_TRUE(isPluginCompilerPreferred(platform, driverVersion, false));
            EXPECT_TRUE(isPluginCompilerPreferred(platform, driverVersion, true));
        }
    }
}

TEST(CompilerSelectionTests, RejectsNonCanonicalPlatforms) {
    const std::array<std::string_view, 9> platforms{{
        "",
        ov::intel_npu::Platform::AUTO_DETECT,
        "unknown",
        "372",
        "37200",
        "NPU3720",
        "VPU3720",
        "400",
        "4001",
    }};

    for (const auto platform : platforms) {
        EXPECT_FALSE(isPluginCompilerPreferred(platform, std::nullopt, false));
        EXPECT_FALSE(isPluginCompilerPreferred(platform, std::nullopt, true));
    }
}

TEST(CompilerSelectionTests, RejectsEmbeddedNullEvenWhenPrefixIsCanonical) {
    constexpr std::string_view platformWithEmbeddedNull{"3720\0suffix", 11};

    EXPECT_FALSE(isPluginCompilerPreferred(platformWithEmbeddedNull, std::nullopt, false));
    EXPECT_FALSE(isPluginCompilerPreferred(platformWithEmbeddedNull, 2565u, true));
}

}  // namespace
