// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "adaptive/headroom.hpp"
#include "adaptive/instance_stats.hpp"
#include "adaptive/period_estimator.hpp"
#include "adaptive/refresh_policy.hpp"
#include "adaptive/signal_hub.hpp"
#include "adaptive/switch_policy.hpp"

using namespace ov::auto_plugin;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;

namespace {
// a base far from the epoch, which the policies use as "never"
const Time base = Time{} + std::chrono::hours(1);

Time at(milliseconds offset) {
    return base + std::chrono::duration_cast<Time::duration>(offset);
}

InstanceStatsSnapshot stats_of(const char* device,
                               microseconds baseline,
                               microseconds recent_mean,
                               size_t samples = 64) {
    InstanceStatsSnapshot snapshot;
    InstanceLatency latency;
    latency.baseline = baseline;
    latency.recent = recent_mean;
    latency.recent_mean = recent_mean;
    latency.samples = samples;
    snapshot.instances.emplace(ModelInstanceKey(device), latency);
    snapshot.active = ModelInstanceKey(device);
    return snapshot;
}
}  // namespace

TEST(PeriodEstimatorTest, measuresARegularRefreshWithJitter) {
    PeriodEstimator estimator;
    EXPECT_FALSE(estimator.period().has_value());
    const int jitter[] = {3, -4, 6, -2, 5, -6, 1, 4};
    for (int i = 0; i < 8; i++) {
        estimator.on_change(at(milliseconds(127 * i + jitter[i])), true);
    }
    ASSERT_TRUE(estimator.period().has_value());
    EXPECT_NEAR(static_cast<double>(estimator.period()->count()), 127.0, 6.0);
}

TEST(PeriodEstimatorTest, needsSeveralGapsBeforeItReportsAPeriod) {
    PeriodEstimator estimator;
    for (int i = 0; i < 3; i++) {
        estimator.on_change(at(milliseconds(100 * i)), true);
    }
    EXPECT_FALSE(estimator.period().has_value());
    estimator.on_change(at(milliseconds(300)), true);
    EXPECT_TRUE(estimator.period().has_value());
}

TEST(PeriodEstimatorTest, impreciseDetectionsAreNotUsed) {
    PeriodEstimator estimator;
    for (int i = 0; i < 10; i++) {
        estimator.on_change(at(milliseconds(100 * i)), false);
    }
    EXPECT_FALSE(estimator.period().has_value());
}

TEST(PeriodEstimatorTest, irregularGapsGiveNoPeriod) {
    PeriodEstimator estimator;
    const int gaps[] = {40, 400, 90, 700, 60, 500, 150};
    int t = 0;
    estimator.on_change(at(milliseconds(t)), true);
    for (const int gap : gaps) {
        t += gap;
        estimator.on_change(at(milliseconds(t)), true);
    }
    EXPECT_FALSE(estimator.reliable());
}

TEST(PeriodEstimatorTest, aSkippedRefreshIsDividedDown) {
    PeriodEstimator estimator;
    for (int i = 0; i < 6; i++) {
        estimator.on_change(at(milliseconds(100 * i)), true);
    }
    // three refreshes passed between the last two detections
    estimator.on_change(at(milliseconds(800)), true, 4);
    ASSERT_TRUE(estimator.period().has_value());
    EXPECT_NEAR(static_cast<double>(estimator.period()->count()), 100.0, 3.0);
}

TEST(PeriodEstimatorTest, lostHeartbeatHasNoPeriodUntilItIsSeenAgain) {
    PeriodEstimator estimator;
    for (int i = 0; i < 6; i++) {
        estimator.on_change(at(milliseconds(100 * i)), true);
    }
    estimator.on_lost();
    EXPECT_FALSE(estimator.period().has_value());
    EXPECT_TRUE(estimator.lost());
    estimator.on_change(at(milliseconds(5000)), true);
    EXPECT_FALSE(estimator.lost());
    EXPECT_TRUE(estimator.period().has_value());
}

TEST(PeriodEstimatorTest, aWrongPeriodIsMeasuredAgain) {
    PeriodEstimator estimator;
    for (int i = 0; i < 6; i++) {
        estimator.on_change(at(milliseconds(100 * i)), true);
    }
    // the backend now refreshes every 300 ms
    for (int i = 0; i < 20; i++) {
        estimator.on_change(at(milliseconds(5000 + 300 * i)), true);
    }
    ASSERT_TRUE(estimator.period().has_value());
    EXPECT_NEAR(static_cast<double>(estimator.period()->count()), 300.0, 10.0);
}

TEST(PeriodEstimatorTest, aLongGapIsNotDividedWhenNoRefreshWasSkippedOnPurpose) {
    PeriodEstimator estimator;
    for (int i = 0; i < 6; i++) {
        estimator.on_change(at(milliseconds(100 * i)), true);
    }
    estimator.on_change(at(milliseconds(800)), true, 1);
    EXPECT_NEAR(static_cast<double>(estimator.period()->count()), 100.0, 3.0);
    // the gap was rejected: it is not three refreshes
    EXPECT_FALSE(estimator.lost());
}

TEST(RefreshPolicyTest, aCheapQueryKeepsTheDataFreshEveryPeriod) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Locked;
    input.period = milliseconds(127);
    input.last_change = at(milliseconds(1000));
    input.query_cost = microseconds(1000);
    input.stats = stats_of("NPU", milliseconds(200), milliseconds(200));
    const auto decision = policy.decide(input, state, at(milliseconds(1010)));
    EXPECT_EQ(decision.mode, RefreshDecision::Mode::Fresh);
    EXPECT_EQ(decision.backoff, 1u);
    // one period after the last refresh, a fifth of a period early
    EXPECT_EQ(decision.next_query, at(milliseconds(1000 + 127)) - std::chrono::microseconds(25400));
    EXPECT_EQ(decision.interval, milliseconds(127));
}

TEST(RefreshPolicyTest, aQueryAsDearAsAnInferenceIsBackedOffWhileQuiet) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Locked;
    input.period = milliseconds(100);
    input.last_change = at(milliseconds(1000));
    input.query_cost = microseconds(2600);
    input.stats = stats_of("NPU", microseconds(1450), microseconds(1450));
    std::vector<unsigned> backoffs;
    for (int i = 0; i < 12; i++) {
        backoffs.push_back(policy.decide(input, state, at(milliseconds(1010))).backoff);
    }
    EXPECT_EQ(backoffs.front(), 1u);
    EXPECT_EQ(backoffs.back(), 4u);
    EXPECT_TRUE(std::is_sorted(backoffs.begin(), backoffs.end()));
    const auto last = policy.decide(input, state, at(milliseconds(1010)));
    EXPECT_EQ(last.mode, RefreshDecision::Mode::Economy);
    EXPECT_EQ(last.interval, milliseconds(400));
}

TEST(RefreshPolicyTest, aSlowCurrentDeviceBringsTheQueriesBack) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Locked;
    input.period = milliseconds(100);
    input.last_change = at(milliseconds(1000));
    input.query_cost = microseconds(2600);
    input.stats = stats_of("NPU", microseconds(1450), microseconds(1450));
    for (int i = 0; i < 12; i++) {
        policy.decide(input, state, at(milliseconds(1010)));
    }
    ASSERT_EQ(state.backoff, 4u);
    input.stats = stats_of("NPU", microseconds(1450), microseconds(3000));
    EXPECT_EQ(policy.decide(input, state, at(milliseconds(1010))).backoff, 1u);
}

TEST(RefreshPolicyTest, aSwitchAndAUtilizationNearItsThresholdBringTheQueriesBack) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Locked;
    input.period = milliseconds(100);
    input.last_change = at(milliseconds(1000));
    input.query_cost = microseconds(2600);
    input.stats = stats_of("NPU", microseconds(1450), microseconds(1450));
    for (int i = 0; i < 12; i++) {
        policy.decide(input, state, at(milliseconds(1010)));
    }
    ASSERT_EQ(state.backoff, 4u);
    input.headroom = 5.f;
    EXPECT_EQ(policy.decide(input, state, at(milliseconds(1010))).backoff, 1u);
    input.headroom = 60.f;
    for (int i = 0; i < 12; i++) {
        policy.decide(input, state, at(milliseconds(1010)));
    }
    ASSERT_EQ(state.backoff, 4u);
    input.stats = stats_of("GPU.0", microseconds(1450), microseconds(1450));
    EXPECT_EQ(policy.decide(input, state, at(milliseconds(1010))).backoff, 1u);
}

TEST(RefreshPolicyTest, withoutAHeartbeatTheDefaultPeriodAndASmallerBackoffApply) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Lost;
    input.query_cost = microseconds(2600);
    input.stats = stats_of("NPU", microseconds(1450), microseconds(1450));
    RefreshDecision decision;
    for (int i = 0; i < 12; i++) {
        decision = policy.decide(input, state, at(milliseconds(1000)));
    }
    EXPECT_EQ(decision.backoff, 2u);
    EXPECT_EQ(decision.interval, milliseconds(2000));
    EXPECT_EQ(decision.next_query, at(milliseconds(3000)));
}

TEST(RefreshPolicyTest, whileProbingTheQueriesAreClose) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Probing;
    const auto decision = policy.decide(input, state, at(milliseconds(1000)));
    EXPECT_EQ(decision.next_query, at(milliseconds(1010)));
}

TEST(RefreshPolicyTest, withoutInferenceStatsTheDataIsKeptFresh) {
    TelemetryRefreshPolicy policy;
    RefreshState state;
    RefreshInput input;
    input.heartbeat = Heartbeat::Locked;
    input.period = milliseconds(100);
    input.last_change = at(milliseconds(1000));
    input.query_cost = microseconds(5000);
    EXPECT_EQ(policy.decide(input, state, at(milliseconds(1010))).mode, RefreshDecision::Mode::Fresh);
}

TEST(SwitchPolicyTest, staysOnTheSameDeviceAndWithinTheDwellTime) {
    SwitchPolicy policy;
    SwitchInput input;
    input.current = "NPU.4000";
    input.proposed = "NPU.4000";
    input.now = at(milliseconds(5000));
    input.current_since = at(milliseconds(0));
    EXPECT_TRUE(policy.decide(input).stay);
    input.proposed = "GPU.0";
    input.current_since = at(milliseconds(4500));
    EXPECT_TRUE(policy.decide(input).stay);
}

TEST(SwitchPolicyTest, leavesAfterTheDwellTimeOrWhenItHasTo) {
    SwitchPolicy policy;
    SwitchInput input;
    input.current = "NPU.4000";
    input.proposed = "GPU.0";
    input.now = at(milliseconds(5000));
    input.current_since = at(milliseconds(3500));
    EXPECT_FALSE(policy.decide(input).stay);
    input.current_since = at(milliseconds(4900));
    EXPECT_TRUE(policy.decide(input).stay);
    input.current_over_threshold = true;
    EXPECT_FALSE(policy.decide(input).stay);
    input.current_over_threshold = false;
    input.current_available = false;
    EXPECT_FALSE(policy.decide(input).stay);
}

TEST(SwitchPolicyTest, aFirstSelectionIsNotHeld) {
    SwitchPolicy policy;
    SwitchInput input;
    input.proposed = "NPU.4000";
    input.now = at(milliseconds(5000));
    EXPECT_TRUE(policy.decide(input).stay);
}

TEST(HeadroomTest, usesTheExactNameBeforeTheBaseName) {
    const std::unordered_map<std::string, unsigned> thresholds{{"GPU", 80}, {"GPU.1", 50}, {"NPU", 90}};
    EXPECT_EQ(find_utilization_threshold(thresholds, "GPU.0"), 80u);
    EXPECT_EQ(find_utilization_threshold(thresholds, "GPU.1"), 50u);
    EXPECT_EQ(find_utilization_threshold(thresholds, "NPU.4000"), 90u);
    EXPECT_FALSE(find_utilization_threshold(thresholds, "CPU").has_value());
}

TEST(HeadroomTest, isTheDistanceOfTheClosestUtilization) {
    const std::unordered_map<std::string, unsigned> thresholds{{"GPU", 80}, {"NPU", 90}};
    const Utilizations utilizations{{"GPU.0", 30.f}, {"NPU.4000", 85.f}, {"CPU", 99.f}};
    EXPECT_FLOAT_EQ(*threshold_headroom(utilizations, thresholds), 5.f);
    EXPECT_FLOAT_EQ(*threshold_headroom({{"GPU.0", 95.f}}, thresholds), -15.f);
    EXPECT_FALSE(threshold_headroom({{"CPU", 50.f}}, thresholds).has_value());
    EXPECT_FALSE(threshold_headroom(utilizations, {}).has_value());
}

TEST(InstanceStatsRecentMeanTest, followsTheLastSamples) {
    InstanceStats stats(32);
    for (int i = 0; i < 24; i++) {
        stats.record_inference("NPU", microseconds(1000));
    }
    for (int i = 0; i < 8; i++) {
        stats.record_inference("NPU", microseconds(3000));
    }
    const auto latency = stats.snapshot().instances.at(ModelInstanceKey("NPU"));
    EXPECT_EQ(latency.recent_mean, microseconds(3000));
    EXPECT_EQ(latency.baseline, microseconds(1000));
}

namespace {
// A backend that refreshes every `period` and takes `cost` to answer, on a clock the test drives.
class FakeBackend : public ISignalSource {
public:
    FakeBackend(Time& clock, milliseconds period, microseconds cost, bool constant = false)
        : m_clock(clock),
          m_start(clock),
          m_period(period),
          m_cost(cost),
          m_constant(constant) {}
    Utilizations query(const DeviceQuery&) override {
        calls++;
        m_clock += std::chrono::duration_cast<Time::duration>(m_cost);
        const auto refreshes = m_constant ? 0 : (m_clock - m_start) / m_period;
        return {{"NPU.4000", static_cast<float>(refreshes % 50)}};
    }
    size_t calls = 0;

private:
    Time& m_clock;
    Time m_start;
    milliseconds m_period;
    microseconds m_cost;
    bool m_constant;
};

struct Replay {
    Time clock = base;
    std::shared_ptr<FakeBackend> backend;
    std::shared_ptr<SignalHub> hub;
    std::shared_ptr<SignalHub::Subscription> subscription;
    size_t reads = 0;
    size_t served = 0;

    explicit Replay(milliseconds period, microseconds cost, bool constant = false, bool with_stats = true) {
        backend = std::make_shared<FakeBackend>(clock, period, cost, constant);
        SignalHub::Options options;
        options.start_thread = false;
        options.clock = [this] {
            return clock;
        };
        hub = std::make_shared<SignalHub>(backend, options);
        SignalHub::StatsFn stats;
        if (with_stats) {
            stats = [] {
                return stats_of("NPU.4000", microseconds(1450), microseconds(1450));
            };
        }
        subscription = hub->subscribe({{"NPU.4000", "npu"}}, stats);
    }

    // an inference every `inference` that reads the utilization, the hub stepping whenever it is due
    void run(milliseconds duration, microseconds inference) {
        const auto end = base + duration;
        Time next_inference = clock;
        Time next_step = clock;
        while (clock < end) {
            const auto now = std::min(next_inference, next_step);
            clock = std::max(clock, now);
            if (next_step <= clock) {
                next_step = hub->step(clock);
            }
            if (next_inference <= clock) {
                reads++;
                served += hub->read({"NPU.4000"}) ? 1 : 0;
                next_inference += std::chrono::duration_cast<Time::duration>(inference);
            }
            if (next_step == SignalHub::never()) {
                next_step = next_inference;
            }
        }
    }
};
}  // namespace

TEST(SignalHubReplayTest, measuresThePeriodAndQueriesFarLessThanOncePerInference) {
    Replay replay(milliseconds(127), microseconds(2600));
    replay.run(seconds(10), microseconds(1500));
    const auto observation = replay.hub->read({"NPU.4000"});
    ASSERT_TRUE(observation.has_value());
    ASSERT_TRUE(observation->period.has_value());
    EXPECT_NEAR(static_cast<double>(observation->period->count()), 127.0, 6.0);
    GTEST_LOG_(INFO) << "queries " << replay.backend->calls << " reads " << replay.reads << " served " << replay.served
                     << " period " << observation->period->count() << " ms";
    // an inference every 1.5 ms would be about 6600 queries
    EXPECT_GT(replay.reads, 6000u);
    EXPECT_LT(replay.backend->calls, 700u);
    // all but the first few reads are answered by the hub
    EXPECT_GT(replay.served, replay.reads * 98 / 100);
}

TEST(SignalHubReplayTest, aCheapQueryIsMadeOncePerPeriod) {
    Replay replay(milliseconds(100), microseconds(200));
    // inferences of 200 ms, so a 200 us query is a thousandth of one
    replay.hub.reset();
    replay.subscription.reset();
    SignalHub::Options options;
    options.start_thread = false;
    options.clock = [&replay] {
        return replay.clock;
    };
    replay.hub = std::make_shared<SignalHub>(replay.backend, options);
    replay.subscription = replay.hub->subscribe({{"NPU.4000", "npu"}}, [] {
        return stats_of("NPU.4000", milliseconds(200), milliseconds(200));
    });
    const auto before = replay.backend->calls;
    replay.run(seconds(20), milliseconds(200));
    const auto queries = replay.backend->calls - before;
    GTEST_LOG_(INFO) << "queries " << queries << " reads " << replay.reads << " served " << replay.served;
    // once the period is known, about one query (and a few around the refresh) per 100 ms
    EXPECT_GT(queries, 100u);
    EXPECT_LT(queries, 1500u);
    EXPECT_GT(replay.served, replay.reads * 9 / 10);
}

TEST(SignalHubReplayTest, withoutARegularRefreshItFallsBackToTheDefaultPeriod) {
    Replay replay(milliseconds(127), microseconds(2600), /*constant=*/true);
    replay.run(seconds(12), microseconds(1500));
    const auto observation = replay.hub->read({"NPU.4000"});
    ASSERT_TRUE(observation.has_value());
    EXPECT_FALSE(observation->period.has_value());
    // probing for 3 s at 10 ms, then a query about every 1-2 s
    EXPECT_LT(replay.backend->calls, 450u);
}

TEST(SignalHubReplayTest, aSlowQueryIsStillAnsweredFromTheLastObservation) {
    Replay replay(milliseconds(127), microseconds(2600));
    replay.run(seconds(5), microseconds(1500));
    const auto observation = replay.hub->read({"NPU.4000"});
    ASSERT_TRUE(observation.has_value());
    EXPECT_LE(replay.clock - observation->observed_at, std::chrono::milliseconds(1200));
    EXPECT_GE(observation->version, 10u);
}

TEST(SignalHubTest, readsWantADeviceTheSubscriptionsCover) {
    Replay replay(milliseconds(127), microseconds(2600));
    replay.run(seconds(2), microseconds(1500));
    EXPECT_TRUE(replay.hub->read({"NPU.4000"}).has_value());
    EXPECT_FALSE(replay.hub->read({"GPU.0"}).has_value());
}

TEST(SignalHubTest, anOldObservationIsNotHandedOut) {
    Replay replay(milliseconds(127), microseconds(2600));
    replay.run(seconds(2), microseconds(1500));
    ASSERT_TRUE(replay.hub->read({"NPU.4000"}).has_value());
    EXPECT_FALSE(replay.hub->read({"NPU.4000"}, replay.clock + std::chrono::seconds(30)).has_value());
}

TEST(SignalHubTest, pausesWithoutReadsAndResumesOnTheNextOne) {
    Replay replay(milliseconds(127), microseconds(2600));
    replay.run(seconds(2), microseconds(1500));
    const auto calls = replay.backend->calls;
    replay.clock += std::chrono::seconds(10);
    EXPECT_EQ(replay.hub->step(replay.clock), SignalHub::never());
    EXPECT_EQ(replay.backend->calls, calls);
    // the first read after the pause is not answered, the caller queries itself, and the hub is due again
    EXPECT_FALSE(replay.hub->read({"NPU.4000"}).has_value());
    EXPECT_NE(replay.hub->step(replay.clock), SignalHub::never());
    EXPECT_GT(replay.backend->calls, calls);
}

TEST(SignalHubTest, withoutSubscriptionsThereIsNothingToQuery) {
    Replay replay(milliseconds(127), microseconds(2600));
    replay.subscription.reset();
    EXPECT_EQ(replay.hub->step(replay.clock), SignalHub::never());
    EXPECT_EQ(replay.backend->calls, 0u);
    EXPECT_FALSE(replay.hub->read({"NPU.4000"}).has_value());
}

TEST(SignalHubTest, endingASubscriptionStopsItsCallbacks) {
    Time clock = base;
    auto backend = std::make_shared<FakeBackend>(clock, milliseconds(127), microseconds(2600));
    SignalHub::Options options;
    options.start_thread = false;
    options.clock = [&clock] {
        return clock;
    };
    auto hub = std::make_shared<SignalHub>(backend, options);
    int stats_calls = 0;
    auto subscription = hub->subscribe({{"NPU.4000", "npu"}}, [&stats_calls] {
        stats_calls++;
        return InstanceStatsSnapshot{};
    });
    hub->step(clock);
    EXPECT_EQ(stats_calls, 1);
    subscription.reset();
    clock += std::chrono::seconds(1);
    hub->step(clock);
    EXPECT_EQ(stats_calls, 1);
}

TEST(SignalHubTest, theThreadQueriesAndStopsWithTheHub) {
    auto backend_clock = std::make_shared<Time>(std::chrono::steady_clock::now());
    class Backend : public ISignalSource {
    public:
        Utilizations query(const DeviceQuery&) override {
            calls++;
            return {{"NPU.4000", static_cast<float>(calls.load())}};
        }
        std::atomic<size_t> calls{0};
    };
    auto backend = std::make_shared<Backend>();
    std::shared_ptr<SignalHub::Subscription> subscription;
    {
        auto hub = std::make_shared<SignalHub>(backend);
        subscription = hub->subscribe({{"NPU.4000", "npu"}}, {});
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (backend->calls < 5 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(milliseconds(5));
        }
        EXPECT_GE(backend->calls.load(), 5u);
        EXPECT_TRUE(hub->read({"NPU.4000"}).has_value());
    }
    // the hub is gone with its thread, and the subscription outlives it without harm
    const auto calls = backend->calls.load();
    std::this_thread::sleep_for(milliseconds(50));
    EXPECT_EQ(backend->calls.load(), calls);
    subscription.reset();
}
