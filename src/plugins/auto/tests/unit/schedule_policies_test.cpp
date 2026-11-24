// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "schedule_policies.hpp"

#include <gtest/gtest.h>

using ov::auto_plugin::CompileStats;
using ov::auto_plugin::FailurePolicy;
using ov::auto_plugin::InstanceUsage;
using ov::auto_plugin::ModelInstanceKey;
using ov::auto_plugin::ReloadablePolicy;
using ov::auto_plugin::Time;

namespace {
InstanceUsage usage(const std::string& device,
                    bool reloadable,
                    Time last_used,
                    std::optional<CompileStats> stats = {}) {
    InstanceUsage result;
    result.instance = ModelInstanceKey(device);
    result.reloadable = reloadable;
    result.last_used = last_used;
    result.stats = std::move(stats);
    return result;
}

CompileStats compiled(size_t count, std::optional<bool> loaded_from_cache) {
    CompileStats stats;
    stats.compile_count = count;
    stats.last_loaded_from_cache = loaded_from_cache;
    return stats;
}
}  // namespace

TEST(ReloadablePolicyTest, releasesEveryReloadableInactiveInstanceByDefault) {
    const auto now = std::chrono::steady_clock::now();
    const auto decision = ReloadablePolicy().decide({usage("GPU.0", true, now), usage("CPU", true, now)}, now);
    EXPECT_EQ(decision.release.size(), 2u);
    EXPECT_TRUE(decision.keep_resident.empty());
}

TEST(ReloadablePolicyTest, keepsAnInstanceThatCannotBeReloaded) {
    const auto now = std::chrono::steady_clock::now();
    const auto decision = ReloadablePolicy().decide({usage("GPU.0", false, now), usage("CPU", true, now)}, now);
    ASSERT_EQ(decision.keep_resident.size(), 1u);
    EXPECT_EQ(decision.keep_resident.front(), ModelInstanceKey("GPU.0"));
    ASSERT_EQ(decision.release.size(), 1u);
    EXPECT_EQ(decision.release.front(), ModelInstanceKey("CPU"));
}

TEST(ReloadablePolicyTest, keepsAnInstanceRecompiledWithoutCacheHit) {
    const auto now = std::chrono::steady_clock::now();
    const ReloadablePolicy policy;
    // the first compilation never hits a cache, it is the recompilation that tells
    EXPECT_EQ(policy.decide({usage("CPU", true, now, compiled(1, false))}, now).release.size(), 1u);
    EXPECT_EQ(policy.decide({usage("CPU", true, now, compiled(2, false))}, now).keep_resident.size(), 1u);
    EXPECT_EQ(policy.decide({usage("CPU", true, now, compiled(2, true))}, now).release.size(), 1u);
    EXPECT_EQ(policy.decide({usage("CPU", true, now, compiled(2, std::nullopt))}, now).release.size(), 1u);
}

TEST(ReloadablePolicyTest, keepsTheMostRecentlyUsedInstancesWithinTheLimit) {
    const auto now = std::chrono::steady_clock::now();
    const ReloadablePolicy policy(1);
    const auto decision = policy.decide({usage("GPU.0", true, now - std::chrono::seconds(2)),
                                         usage("GPU.1", true, now - std::chrono::seconds(1)),
                                         usage("CPU", true, now - std::chrono::seconds(3))},
                                        now);
    ASSERT_EQ(decision.release.size(), 2u);
    EXPECT_EQ(decision.release[0], ModelInstanceKey("GPU.0"));
    EXPECT_EQ(decision.release[1], ModelInstanceKey("CPU"));
}

TEST(ReloadablePolicyTest, keepsAnInstanceUsedWithinTheMinimumIdleTime) {
    const auto now = std::chrono::steady_clock::now();
    const ReloadablePolicy policy(0, std::chrono::milliseconds(500));
    const auto decision = policy.decide(
        {usage("GPU.0", true, now - std::chrono::milliseconds(100)), usage("CPU", true, now - std::chrono::seconds(1))},
        now);
    ASSERT_EQ(decision.release.size(), 1u);
    EXPECT_EQ(decision.release.front(), ModelInstanceKey("CPU"));
}

TEST(FailurePolicyTest, backsOffLongerAfterEachFailure) {
    const auto now = std::chrono::steady_clock::now();
    FailurePolicy policy(std::chrono::milliseconds(100), 3);
    EXPECT_TRUE(policy.is_available("GPU.0", now));
    EXPECT_FALSE(policy.record_failure("GPU.0", now));
    EXPECT_FALSE(policy.is_available("GPU.0", now + std::chrono::milliseconds(99)));
    EXPECT_TRUE(policy.is_available("GPU.0", now + std::chrono::milliseconds(100)));
    EXPECT_FALSE(policy.record_failure("GPU.0", now));
    EXPECT_FALSE(policy.is_available("GPU.0", now + std::chrono::milliseconds(199)));
    EXPECT_TRUE(policy.is_available("GPU.0", now + std::chrono::milliseconds(200)));
    EXPECT_TRUE(policy.is_available("CPU", now));
}

TEST(FailurePolicyTest, givesUpAfterTooManyFailures) {
    const auto now = std::chrono::steady_clock::now();
    FailurePolicy policy(std::chrono::milliseconds(1), 2);
    EXPECT_FALSE(policy.record_failure("GPU.0", now));
    EXPECT_TRUE(policy.record_failure("GPU.0", now));
}

TEST(FailurePolicyTest, successClearsTheFailures) {
    const auto now = std::chrono::steady_clock::now();
    FailurePolicy policy(std::chrono::seconds(10), 2);
    EXPECT_FALSE(policy.record_failure("GPU.0", now));
    policy.record_success("GPU.0");
    EXPECT_TRUE(policy.is_available("GPU.0", now));
    EXPECT_FALSE(policy.record_failure("GPU.0", now));
}
