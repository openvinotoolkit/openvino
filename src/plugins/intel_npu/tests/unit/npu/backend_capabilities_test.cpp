// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/backend_capabilities.hpp"

#include <gtest/gtest.h>

#include "intel_npu/common/npu.hpp"
#include "zero_backend_capabilities.hpp"

using intel_npu::BackendCapabilities;
using intel_npu::makeBackendCapabilities;
using intel_npu::ZeroCapabilities;

namespace {

// Minimal backend that implements only what is mandatory, to check the defaults of the interface
class MinimalBackend final : public intel_npu::IEngineBackend {
public:
    const std::string getName() const override {
        return "MINIMAL";
    }
    void updateInfo(const ov::AnyMap&) override {}
};

}  // namespace

TEST(BackendCapabilities, DefaultConstructedReportsNothingAvailable) {
    const BackendCapabilities caps;

    EXPECT_EQ(caps.driverVersion, 0u);
    EXPECT_EQ(caps.graphExtVersion, 0u);
    EXPECT_EQ(caps.commandQueueExtVersion, 0u);
    EXPECT_EQ(caps.apiVersion, 0u);
    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_FALSE(caps.contextExtSupported);
}

TEST(BackendCapabilities, BackendThatDoesNotOverrideReportsNothingAvailable) {
    const MinimalBackend backend;
    const auto caps = backend.getCapabilities();

    EXPECT_EQ(caps.driverVersion, 0u);
    EXPECT_EQ(caps.graphExtVersion, 0u);
    EXPECT_EQ(caps.commandQueueExtVersion, 0u);
    EXPECT_EQ(caps.apiVersion, 0u);
    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_FALSE(caps.contextExtSupported);
}

TEST(MakeBackendCapabilities, EmptyZeroCapabilitiesGiveNothingAvailable) {
    const auto caps = makeBackendCapabilities(ZeroCapabilities{});

    EXPECT_EQ(caps.driverVersion, 0u);
    EXPECT_EQ(caps.graphExtVersion, 0u);
    EXPECT_EQ(caps.commandQueueExtVersion, 0u);
    EXPECT_EQ(caps.apiVersion, 0u);
    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_FALSE(caps.contextExtSupported);
}

TEST(MakeBackendCapabilities, VersionsAreCopied) {
    ZeroCapabilities zeroCaps;
    zeroCaps.driverVersion = 1234;
    zeroCaps.graphExtVersion = ZE_MAKE_VERSION(1, 16);
    zeroCaps.commandQueueExtVersion = ZE_MAKE_VERSION(1, 1);
    zeroCaps.zeApiVersion = ZE_API_VERSION_1_18;

    const auto caps = makeBackendCapabilities(zeroCaps);

    EXPECT_EQ(caps.driverVersion, 1234u);
    EXPECT_EQ(caps.graphExtVersion, static_cast<uint32_t>(ZE_MAKE_VERSION(1, 16)));
    EXPECT_EQ(caps.commandQueueExtVersion, static_cast<uint32_t>(ZE_MAKE_VERSION(1, 1)));
    EXPECT_EQ(caps.apiVersion, static_cast<uint32_t>(ZE_MAKE_VERSION(1, 18)));
}

TEST(MakeBackendCapabilities, SupportFlagsAreCopiedIndependently) {
    ZeroCapabilities onlyLuid;
    onlyLuid.luidExtSupported = true;
    auto caps = makeBackendCapabilities(onlyLuid);

    EXPECT_TRUE(caps.luidExtSupported);
    EXPECT_FALSE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.contextExtSupported);

    ZeroCapabilities onlyContext;
    onlyContext.contextExtSupported = true;
    caps = makeBackendCapabilities(onlyContext);

    EXPECT_TRUE(caps.contextExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_FALSE(caps.commandQueueExtSupported);

    ZeroCapabilities onlyQueue;
    onlyQueue.commandQueueExtSupported = true;
    caps = makeBackendCapabilities(onlyQueue);

    EXPECT_TRUE(caps.commandQueueExtSupported);
    EXPECT_FALSE(caps.luidExtSupported);
    EXPECT_FALSE(caps.contextExtSupported);
}

TEST(MakeBackendCapabilities, ValuesOnlyUsedByLevelZeroComponentsAreNotExposed) {
    // These live in ZeroCapabilities for the level zero components, the plugin has no use for them. Changing them must
    // not change anything the plugin sees.
    ZeroCapabilities baseline;
    ZeroCapabilities different;
    different.commandQueueGroupOrdinal = 5;
    different.mutableCommandListExtVersion = ZE_MAKE_VERSION(1, 1);
    different.commandQueueSetPrioritySupported = true;
    different.externalMemoryStandardAllocationSupported = true;
    different.externalMemoryFdWin32Supported = true;

    const auto a = makeBackendCapabilities(baseline);
    const auto b = makeBackendCapabilities(different);

    EXPECT_EQ(a.driverVersion, b.driverVersion);
    EXPECT_EQ(a.graphExtVersion, b.graphExtVersion);
    EXPECT_EQ(a.commandQueueExtVersion, b.commandQueueExtVersion);
    EXPECT_EQ(a.apiVersion, b.apiVersion);
    EXPECT_EQ(a.commandQueueExtSupported, b.commandQueueExtSupported);
    EXPECT_EQ(a.luidExtSupported, b.luidExtSupported);
    EXPECT_EQ(a.contextExtSupported, b.contextExtSupported);
}
