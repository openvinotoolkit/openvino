// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <thread>

#include "adaptive/decision_log.hpp"
#include "adaptive/goal.hpp"
#include "adaptive/instance_stats.hpp"
#include "plugin_config.hpp"

using namespace ov::auto_plugin;
using std::chrono::microseconds;

TEST(InstanceStatsTest, emptyWithoutSamples) {
    InstanceStats stats;
    EXPECT_TRUE(stats.snapshot().instances.empty());
    EXPECT_FALSE(stats.last_inference(ModelInstanceKey("NPU")).has_value());
}

TEST(InstanceStatsTest, baselineIsALowPercentileAndRecentIsTheLastSample) {
    InstanceStats stats(20);
    for (int i = 1; i <= 20; i++) {
        stats.record_inference("NPU", microseconds(i * 100));
    }
    stats.record_inference("NPU", microseconds(5000));
    const auto latency = stats.snapshot().instances.at(ModelInstanceKey("NPU"));
    EXPECT_EQ(latency.samples, 20u);
    EXPECT_EQ(latency.total, 21u);
    EXPECT_EQ(latency.recent, microseconds(5000));
    EXPECT_LE(latency.baseline, microseconds(400));
    EXPECT_EQ(*stats.last_inference("NPU"), microseconds(5000));
}

TEST(InstanceStatsTest, oldSamplesLeaveTheWindow) {
    InstanceStats stats(4);
    for (int i = 0; i < 4; i++) {
        stats.record_inference("GPU.0", microseconds(10));
    }
    for (int i = 0; i < 4; i++) {
        stats.record_inference("GPU.0", microseconds(900));
    }
    EXPECT_EQ(stats.snapshot().instances.at(ModelInstanceKey("GPU.0")).baseline, microseconds(900));
}

TEST(InstanceStatsTest, instancesAreSeparateAndTheHelperIsNotTheCpu) {
    InstanceStats stats;
    stats.record_inference("CPU", microseconds(100));
    stats.record_inference(ModelInstanceKey("CPU", ModelInstanceKey::Profile::StartupHelper), microseconds(900));
    stats.set_active(ModelInstanceKey("CPU"));
    const auto snapshot = stats.snapshot();
    EXPECT_EQ(snapshot.instances.size(), 2u);
    EXPECT_EQ(snapshot.active, ModelInstanceKey("CPU"));
    EXPECT_EQ(*stats.last_inference("CPU"), microseconds(100));
}

TEST(InstanceStatsTest, concurrentRecordingKeepsEveryCount) {
    InstanceStats stats;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&stats] {
            for (int i = 0; i < 500; i++) {
                stats.record_inference("NPU", microseconds(100));
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(stats.snapshot().instances.at(ModelInstanceKey("NPU")).total, 2000u);
}

TEST(DecisionLogTest, summaryHasTheTotalsAndOneLinePerInstance) {
    DecisionLog log;
    log.count_inference();
    log.count_inference();
    log.count_switch();
    InstanceStats stats;
    stats.record_inference("NPU", microseconds(1600));
    const auto summary = log.summary(stats.snapshot());
    EXPECT_NE(summary.find("total inference:2, device switch:1"), std::string::npos) << summary;
    EXPECT_NE(summary.find("NPU: inference:1, baseline:1600 us, recent:1600 us"), std::string::npos) << summary;
}

TEST(AdaptiveLevelTest, parsesCaseInsensitively) {
    EXPECT_EQ(parse_adaptive_level("off"), AdaptiveLevel::Off);
    EXPECT_EQ(parse_adaptive_level("Route"), AdaptiveLevel::Route);
    EXPECT_EQ(parse_adaptive_level("FULL"), AdaptiveLevel::Full);
    EXPECT_FALSE(parse_adaptive_level("auto").has_value());
}

TEST(AdaptiveLevelTest, isAPrivatePropertyOffByDefault) {
    PluginConfig config;
    EXPECT_EQ(config.get_property(adaptive_level), "OFF");
    const auto supported = config.supported_properties();
    EXPECT_TRUE(std::none_of(supported.begin(), supported.end(), [](const ov::PropertyName& name) {
        return name == adaptive_level.name();
    }));
    config.set_property(ov::AnyMap{{adaptive_level.name(), "route"}});
    EXPECT_EQ(config.get_property(adaptive_level), "ROUTE");
    EXPECT_ANY_THROW(config.set_property(ov::AnyMap{{adaptive_level.name(), "sometimes"}}));
}
