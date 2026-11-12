// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_capabilities.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

using intel_npu::makeZeroCapabilities;
using intel_npu::ZeroCapabilities;

namespace {

using SupportedExtensions = std::map<std::string, uint32_t, std::less<>>;

ZeroCapabilities build(const SupportedExtensions& extensions,
                       bool standardAllocation = false,
                       bool fdWin32 = false,
                       uint32_t ordinal = 0,
                       ze_api_version_t apiVersion = {}) {
    return makeZeroCapabilities(extensions, standardAllocation, fdWin32, ordinal, apiVersion);
}

}  // namespace

TEST(ZeroCapabilities, DefaultConstructedSupportsNothing) {
    const ZeroCapabilities caps;

    EXPECT_EQ(caps.mutableCommandListExtVersion, 0u);
    EXPECT_EQ(caps.commandQueueGroupOrdinal, 0u);
    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.commandQueueSetPrioritySupported);
    EXPECT_FALSE(caps.contextExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_FALSE(caps.externalMemoryStandardAllocationSupported);
    EXPECT_FALSE(caps.externalMemoryFdWin32Supported);
}

TEST(ZeroCapabilities, NoNegotiatedExtensionsMeansNothingSupported) {
    const auto caps = build({});

    EXPECT_EQ(caps.mutableCommandListExtVersion, 0u);
    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.commandQueueSetPrioritySupported);
    EXPECT_FALSE(caps.contextExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
}

TEST(ZeroCapabilities, EveryNegotiatedExtensionIsReported) {
    const auto caps = build({{ZE_COMMAND_QUEUE_NPU_EXT_NAME, ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_1},
                             {ZE_COMMAND_QUEUE_SET_PRIORITY_EXT_NAME, ZE_COMMAND_QUEUE_SET_PRIORITY_EXT_VERSION_1_0},
                             {ZE_CONTEXT_NPU_EXT_NAME, ZE_CONTEXT_NPU_EXT_VERSION_1_0},
                             {ZE_DEVICE_LUID_EXT_NAME, ZE_DEVICE_LUID_EXT_VERSION_1_0},
                             {ZE_MUTABLE_COMMAND_LIST_EXP_NAME, ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1}});

    EXPECT_TRUE(caps.commandQueueExtSupported);
    EXPECT_TRUE(caps.commandQueueSetPrioritySupported);
    EXPECT_TRUE(caps.contextExtSupported);
    EXPECT_TRUE(caps.luidExtSupported);
    EXPECT_EQ(caps.mutableCommandListExtVersion, static_cast<uint32_t>(ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1));
}

TEST(ZeroCapabilities, ExtensionsAreReportedIndependently) {
    const auto onlyPriority =
        build({{ZE_COMMAND_QUEUE_SET_PRIORITY_EXT_NAME, ZE_COMMAND_QUEUE_SET_PRIORITY_EXT_VERSION_1_0}});

    EXPECT_TRUE(onlyPriority.commandQueueSetPrioritySupported);
    EXPECT_FALSE(onlyPriority.commandQueueExtSupported);
    EXPECT_FALSE(onlyPriority.contextExtSupported);
    EXPECT_FALSE(onlyPriority.luidExtSupported);
    EXPECT_EQ(onlyPriority.mutableCommandListExtVersion, 0u);

    const auto onlyContext = build({{ZE_CONTEXT_NPU_EXT_NAME, ZE_CONTEXT_NPU_EXT_VERSION_1_0}});

    EXPECT_TRUE(onlyContext.contextExtSupported);
    EXPECT_FALSE(onlyContext.commandQueueSetPrioritySupported);
}

TEST(ZeroCapabilities, MutableCommandListKeepsTheNegotiatedVersion) {
    // The driver may only support an older version than the one the plugin targets, the older one must be reported
    const auto older = build({{ZE_MUTABLE_COMMAND_LIST_EXP_NAME, ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_0}});
    const auto newer = build({{ZE_MUTABLE_COMMAND_LIST_EXP_NAME, ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1}});

    EXPECT_EQ(older.mutableCommandListExtVersion, static_cast<uint32_t>(ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_0));
    EXPECT_EQ(newer.mutableCommandListExtVersion, static_cast<uint32_t>(ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1));
    EXPECT_NE(older.mutableCommandListExtVersion, newer.mutableCommandListExtVersion);
}

TEST(ZeroCapabilities, UnrelatedExtensionsAreIgnored) {
    const auto caps = build({{"ZE_some_other_extension", 1}});

    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.commandQueueSetPrioritySupported);
    EXPECT_FALSE(caps.contextExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_EQ(caps.mutableCommandListExtVersion, 0u);
}

TEST(ZeroCapabilities, DiscoveredValuesArePassedThrough) {
    const auto caps = build({}, /*standardAllocation=*/true, /*fdWin32=*/false, /*ordinal=*/3, ZE_API_VERSION_1_18);

    EXPECT_TRUE(caps.externalMemoryStandardAllocationSupported);
    EXPECT_FALSE(caps.externalMemoryFdWin32Supported);
    EXPECT_EQ(caps.commandQueueGroupOrdinal, 3u);
    EXPECT_EQ(caps.zeApiVersion, ZE_API_VERSION_1_18);
}

TEST(ZeroCapabilities, ExternalMemoryFlagsAreIndependent) {
    const auto fdOnly = build({}, false, true);
    const auto stdOnly = build({}, true, false);

    EXPECT_TRUE(fdOnly.externalMemoryFdWin32Supported);
    EXPECT_FALSE(fdOnly.externalMemoryStandardAllocationSupported);
    EXPECT_TRUE(stdOnly.externalMemoryStandardAllocationSupported);
    EXPECT_FALSE(stdOnly.externalMemoryFdWin32Supported);
}

TEST(ZeroCapabilities, IsACopyableValue) {
    ZeroCapabilities original;
    original.commandQueueGroupOrdinal = 7;
    original.luidExtSupported = true;

    ZeroCapabilities copy = original;
    original.commandQueueGroupOrdinal = 1;

    EXPECT_EQ(copy.commandQueueGroupOrdinal, 7u);
    EXPECT_TRUE(copy.luidExtSupported);
}
