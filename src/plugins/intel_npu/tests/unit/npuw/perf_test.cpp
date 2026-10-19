// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "perf.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#include "perf_tags.hpp"

#ifdef NPU_PLUGIN_DEVELOPER_BUILD

namespace {

namespace perf = ov::npuw::perf;

using MS = perf::metric<perf::MSec>;
using MB = perf::metric<perf::Bytes>;

constexpr auto npos = std::string::npos;

class ScopedProfile {
public:
    explicit ScopedProfile(const std::string& area = "test", perf::Scope scope = perf::Scope::Execution)
        : m_profile(scope, true) {
        m_profile.area = area;
    }
    ~ScopedProfile() {
        m_profile.enable(false);
    }

    perf::Profile<MS>* operator->() {
        return &m_profile;
    }
    perf::Profile<MS>& operator*() {
        return m_profile;
    }

private:
    perf::Profile<MS> m_profile;
};

struct RequestGroup {
    ScopedProfile llm;
    ScopedProfile child;
    perf::RunBoundary boundary;

    void snapshot() {
        boundary.snapshot_and_reset({&*llm, &*child});
    }
};

std::string to_string(const MS& m) {
    std::ostringstream os;
    m.print(os, "tag");
    return os.str();
}

}  // namespace

TEST(NpuwPerfMetric, CollectsStatistics) {
    MS m;
    m.add(1.0f);
    m.add(3.0f);
    m.add(2.0f);

    EXPECT_TRUE(m.has_samples());
    EXPECT_EQ(3u, m.count());
    EXPECT_FLOAT_EQ(6.0f, m.sum());
    EXPECT_FLOAT_EQ(2.0f, m.avg());
    EXPECT_FLOAT_EQ(2.0f, m.med());
    ASSERT_EQ(3u, m.samples().size());
    EXPECT_FLOAT_EQ(1.0f, m.samples()[0]);
}

TEST(NpuwPerfMetric, EmptyPrintsNoData) {
    MS m;
    EXPECT_NE(npos, to_string(m).find("[ no data ]"));
}

TEST(NpuwPerfMetric, MoveKeepsStatistics) {
    MS src;
    src.add(1.0f);
    src.add(5.0f);

    MS dst(std::move(src));
    EXPECT_EQ(2u, dst.count());
    EXPECT_FLOAT_EQ(6.0f, dst.sum());
    // vmin/vmax must survive the move too
    EXPECT_NE(npos, to_string(dst).find("1..5"));
}

TEST(NpuwPerfMetric, MaxHandlesNegativeSamples) {
    // vmax must start at lowest(), not min() (which is the smallest _positive_ float)
    MS m;
    m.add(-3.0f);
    m.add(-1.0f);
    EXPECT_NE(npos, to_string(m).find("-3..-1"));
}

TEST(NpuwPerfMetric, BytesHasSamples) {
    MB c;
    EXPECT_FALSE(c.has_samples());
    c.add(1024u * 1024u);
    EXPECT_TRUE(c.has_samples());
    EXPECT_EQ(1u, c.count());
    EXPECT_FLOAT_EQ(1.0f, c.sum_f());
}

TEST(NpuwPerfProfile, HandleIsStableAndUnique) {
    ScopedProfile p;

    auto* a = p->handle("a");
    auto* b = p->handle("b");
    ASSERT_NE(nullptr, a);
    ASSERT_NE(nullptr, b);
    EXPECT_NE(a, b);
    EXPECT_EQ(a, p->handle("a"));

    // Inserting other tags must not invalidate an existing handle
    for (int i = 0; i < 100; i++) {
        p->handle("x" + std::to_string(i));
    }
    EXPECT_EQ(a, p->handle("a"));

    a->add(1.0f);
    p->reset();
    EXPECT_EQ(a, p->handle("a"));
    EXPECT_FALSE(a->has_samples());

    a->add(2.0f);
    p->snapshot_and_reset();
    EXPECT_EQ(a, p->handle("a"));
}

TEST(NpuwPerfProfile, HandleIsNullWhenDisabled) {
    perf::Profile<MS> p(perf::Scope::Execution, false);
    p.area = "disabled";
    EXPECT_EQ(nullptr, p.handle("a"));
    EXPECT_FALSE(p.has_samples());
}

TEST(NpuwPerfProfile, HasSamplesIsNotEmptiness) {
    ScopedProfile p;
    // A pre-resolved handle creates an entry eagerly - metrics.empty() is no longer
    // a valid "nothing was collected" test
    ASSERT_NE(nullptr, p->handle("a"));
    EXPECT_FALSE(p->metrics.empty());
    EXPECT_FALSE(p->has_samples());
}

TEST(NpuwPerfProfile, SnapshotMovesSamplesAndKeepsTags) {
    if (perf::keep_runs() == 0u) {
        GTEST_SKIP() << "OPENVINO_NPUW_PROF_KEEP_RUNS=0 retains no snapshots";
    }
    ScopedProfile p;
    const auto base = std::size_t{0};
    auto* a = p->handle("a");
    a->add(1.0f);
    a->add(2.0f);

    p->snapshot_and_reset();

    ASSERT_EQ(1u, p->snapshots.size());
    EXPECT_EQ(base, p->snapshots.front().index);
    ASSERT_EQ(1u, p->snapshots.front().metrics.count("a"));
    EXPECT_EQ(2u, p->snapshots.front().metrics.at("a").count());

    EXPECT_FALSE(p->has_samples());
    EXPECT_EQ(1u, p->metrics.count("a"));  // tag survives
}

TEST(NpuwPerfProfile, SnapshotSkipsAnEmptyRun) {
    ScopedProfile p;
    p->handle("a");
    p->snapshot_and_reset();
    EXPECT_TRUE(p->snapshots.empty());
}

TEST(NpuwPerfProfile, RetainedRunCapDropsTheOldest) {
    ScopedProfile p;
    const auto base = std::size_t{0};
    auto* a = p->handle("a");

    const auto cap = perf::keep_runs();
    const std::size_t runs = cap + 4u;
    for (std::size_t i = 0; i < runs; i++) {
        a->add(static_cast<float>(i) + 1.0f);
        p->snapshot_and_reset();
    }

    ASSERT_EQ(cap, p->snapshots.size());
    if (cap == 0u) {
        // OPENVINO_NPUW_PROF_KEEP_RUNS=0 means "only the current run matters"
        return;
    }
    EXPECT_EQ(base + runs - cap, p->snapshots.front().index);
}

TEST(NpuwPerfProfile, RunIdIsSharedAcrossProfiles) {
    ScopedProfile busy("busy", perf::Scope::Execution);
    ScopedProfile idle("idle", perf::Scope::Execution);
    perf::RunBoundary boundary;
    const auto ending = boundary.current_run_index();

    // Conversation `ending`: only `busy` collects anything
    busy->handle("a")->add(1.0f);
    boundary.snapshot_and_reset({&*busy, &*idle});

    // Conversation `ending + 1`: only `idle` collects anything
    idle->handle("a")->add(1.0f);

    std::ostringstream busy_os;
    std::ostringstream idle_os;
    busy->report(busy_os);
    idle->report(idle_os);

    // Both profiles must label the new conversation the same way, even though `idle`
    // had nothing to snapshot at the boundary
    EXPECT_NE(npos, idle_os.str().find("run[" + std::to_string(ending + 1u) + "] (current)"));
    EXPECT_EQ(npos, busy_os.str().find("(current)")) << "busy collected nothing after the boundary";
}

TEST(NpuwPerfProfile, DisabledProfileDoesNotAllocateEntries) {
    perf::Profile<MS> p(perf::Scope::Execution, false);
    p.area = "disabled";

    bool ran = false;
    p.record("1/prefill:3b.infer", [&]() {
        ran = true;
    });

    EXPECT_TRUE(ran) << "The timed body must still run";
    EXPECT_TRUE(p.metrics.empty()) << "A disabled profile must not build tags or insert entries";
}

TEST(NpuwPerfProfile, EnabledRecordCollectsUnderTheTag) {
    ScopedProfile p;
    p->record("1/prefill:3b.infer", []() {});
    ASSERT_EQ(1u, p->metrics.count("1/prefill:3b.infer"));
    EXPECT_EQ(1u, p->metrics.at("1/prefill:3b.infer").count());
}

TEST(NpuwPerfProfile, SnapshottedRunIsNotReportedTwice) {
    if (perf::keep_runs() == 0u) {
        GTEST_SKIP() << "OPENVINO_NPUW_PROF_KEEP_RUNS=0 retains no snapshots";
    }
    ScopedProfile p("Model0/performance");
    const auto base = std::size_t{0};
    p->handle("total@CPU")->add(1.0f);
    p->snapshot_and_reset();

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();
    EXPECT_NE(npos, text.find("run[" + std::to_string(base) + "]"));
    EXPECT_EQ(npos, text.find("(current)"));
}

TEST(NpuwPerfBoundary, RunResetOnlyTouchesExecutionScope) {
    ScopedProfile exec("exec", perf::Scope::Execution);
    ScopedProfile comp("compilation", perf::Scope::Compilation);
    ScopedProfile life("memory", perf::Scope::Lifetime);

    exec->handle("a")->add(1.0f);
    comp->handle("a")->add(1.0f);
    life->handle("a")->add(1.0f);

    exec->reset();

    EXPECT_FALSE(exec->has_samples());
    EXPECT_TRUE(comp->has_samples());
    EXPECT_TRUE(life->has_samples());
}

TEST(NpuwPerfBoundary, SnapshotResetOnlyTouchesExecutionScope) {
    if (perf::keep_runs() == 0u) {
        GTEST_SKIP() << "OPENVINO_NPUW_PROF_KEEP_RUNS=0 retains no snapshots";
    }
    ScopedProfile exec("exec", perf::Scope::Execution);
    ScopedProfile comp("compilation", perf::Scope::Compilation);

    exec->handle("a")->add(1.0f);
    comp->handle("a")->add(1.0f);

    perf::RunBoundary boundary;
    boundary.snapshot_and_reset({&*exec});

    EXPECT_EQ(1u, exec->snapshots.size());
    EXPECT_TRUE(comp->snapshots.empty());
    EXPECT_TRUE(comp->has_samples());
}

TEST(NpuwPerfBoundary, DestroyedProfileIsNotVisited) {
    {
        ScopedProfile tmp;
        tmp->handle("a")->add(1.0f);
    }
    // A boundary owns no registry and only visits the explicitly collected profiles.
    perf::RunBoundary boundary;
    boundary.snapshot_and_reset({});
    SUCCEED();
}

TEST(NpuwPerfBoundary, IndependentOwnersOnlySnapshotTheirOwnProfiles) {
    if (perf::keep_runs() == 0u) {
        GTEST_SKIP() << "OPENVINO_NPUW_PROF_KEEP_RUNS=0 retains no snapshots";
    }
    RequestGroup first;
    RequestGroup second;
    first.llm->add("llm", 1.f);
    first.child->add("child", 2.f);
    second.llm->add("llm", 3.f);
    second.child->add("child", 4.f);

    first.snapshot();

    for (auto* profile : {&*first.llm, &*first.child}) {
        ASSERT_EQ(1u, profile->snapshots.size());
        EXPECT_EQ(0u, profile->snapshots.front().index);
        EXPECT_FALSE(profile->has_samples());
    }
    for (auto* profile : {&*second.llm, &*second.child}) {
        EXPECT_TRUE(profile->snapshots.empty());
        EXPECT_TRUE(profile->has_samples());
    }
}

TEST(NpuwPerfBoundary, ConcurrentOtherOwnerKeepsAllSamplesAndTags) {
    RequestGroup first;
    RequestGroup second;
    constexpr int samples = 2000;
    std::atomic<bool> start{false};
    std::thread recorder([&] {
        while (!start.load(std::memory_order_acquire)) {
        }
        for (int i = 0; i < samples; ++i) {
            second.child->add("tag-" + std::to_string(i), 1.f);
        }
    });
    std::thread boundaries([&] {
        start.store(true, std::memory_order_release);
        for (int i = 0; i < samples; ++i) {
            first.llm->add("llm", 1.f);
            first.snapshot();
        }
    });
    recorder.join();
    boundaries.join();

    EXPECT_TRUE(second.child->snapshots.empty());
    ASSERT_EQ(samples, second.child->metrics.size());
    std::size_t total = 0u;
    for (const auto& metric : second.child->metrics) {
        total += metric.second.count();
    }
    EXPECT_EQ(samples, total);
}

TEST(NpuwPerfBoundary, RunIndicesArePerOwnerAndLateProfilesGetNextSnapshotId) {
    if (perf::keep_runs() == 0u) {
        GTEST_SKIP() << "OPENVINO_NPUW_PROF_KEEP_RUNS=0 retains no snapshots";
    }
    RequestGroup first;
    RequestGroup second;
    first.llm->add("llm", 1.f);
    first.snapshot();
    first.llm->add("llm", 1.f);
    first.snapshot();
    second.llm->add("llm", 1.f);
    second.snapshot();

    ASSERT_EQ(2u, first.llm->snapshots.size());
    EXPECT_EQ(0u, first.llm->snapshots[0].index);
    EXPECT_EQ(1u, first.llm->snapshots[1].index);
    ASSERT_EQ(1u, second.llm->snapshots.size());
    EXPECT_EQ(0u, second.llm->snapshots[0].index);

    ScopedProfile late("late");
    late->add("late", 1.f);
    first.boundary.snapshot_and_reset({&*late});
    ASSERT_EQ(1u, late->snapshots.size());
    EXPECT_EQ(2u, late->snapshots.front().index);
    late->add("late", 1.f);
    std::ostringstream os;
    late->report(os);
    EXPECT_NE(npos, os.str().find("run[3] (current)"));
}

TEST(NpuwPerfSample, CommitsOnce) {
    ScopedProfile p;
    auto* a = p->handle("a");
    {
        perf::sample s(a);
        s.commit();
    }
    EXPECT_EQ(1u, a->count());
}

TEST(NpuwPerfSample, DiscardCancelsTheCommit) {
    ScopedProfile p;
    auto* a = p->handle("a");
    {
        perf::sample s(a);
        s.commit();
        s.discard();
    }
    EXPECT_EQ(0u, a->count());
}

TEST(NpuwPerfSample, NoCommitRecordsNothing) {
    ScopedProfile p;
    auto* a = p->handle("a");
    {
        perf::sample s(a);
        (void)s.elapsed_ms();
    }
    EXPECT_EQ(0u, a->count());
}

TEST(NpuwPerfSample, NullHandleIsInert) {
    perf::sample s(nullptr);
    EXPECT_FLOAT_EQ(0.0f, s.elapsed_ms());
    EXPECT_FLOAT_EQ(0.0f, s.commit());
}

TEST(NpuwPerfSample, MovingTransfersThePendingCommit) {
    ScopedProfile p;
    auto* a = p->handle("a");
    {
        perf::sample from(a);
        perf::sample to(std::move(from));
        EXPECT_FLOAT_EQ(0.0f, from.commit());
        to.commit();
    }
    EXPECT_EQ(1u, a->count());
}

TEST(NpuwPerfSample, InvalidCompletionTimestampUsesCurrentTime) {
    ScopedProfile p;
    auto* a = p->handle("a");
    {
        perf::sample s(a);
        s.commit_at(std::chrono::steady_clock::time_point{});
    }
    ASSERT_EQ(1u, a->count());
    EXPECT_GE(a->samples().front(), 0.f);
    EXPECT_LT(a->samples().front(), 1000.f);
}

TEST(NpuwPerfProfile, AddIsGated) {
    perf::Profile<MS> p(perf::Scope::Execution, false);
    p.add("tag", 1.f);
    EXPECT_TRUE(p.metrics.empty());
    p.enable(true);
    p.add("tag", 2.f);
    EXPECT_EQ(1u, p.metrics.size());
    EXPECT_FLOAT_EQ(2.f, p.metrics.at("tag").sum());
    p.enable(false);
}

TEST(NpuwPerfProfile, LazyHandleBuildsTagOnlyOnceWhenEnabled) {
    perf::Profile<MS> p(perf::Scope::Execution, false);
    perf::metric_ptr cached = nullptr;
    int calls = 0;
    auto make_tag = [&]() {
        ++calls;
        return std::string("tag");
    };
    EXPECT_EQ(nullptr, p.handle_lazy(cached, make_tag));
    EXPECT_EQ(0, calls);
    p.enable(true);
    auto* first = p.handle_lazy(cached, make_tag);
    ASSERT_NE(nullptr, first);
    EXPECT_EQ(first, p.handle_lazy(cached, make_tag));
    EXPECT_EQ(1, calls);
    p.enable(false);
}

TEST(NpuwPerfSample, ExceptionUnwindingRecordsNothing) {
    ScopedProfile p;
    auto* a = p->handle("a");
    auto body = [](bool fail) {
        if (fail) {
            throw std::runtime_error("boom");
        }
    };

    try {
        perf::sample s(a);
        body(true);
        s.commit();
    } catch (const std::exception&) {
    }
    EXPECT_EQ(0u, a->count());

    {
        perf::sample s(a);
        body(false);
        s.commit();
    }
    EXPECT_EQ(1u, a->count());
}

TEST(NpuwPerfSample, IsNotCopyableButMovable) {
    static_assert(!std::is_copy_constructible<perf::sample>::value, "sample must not be copyable");
    static_assert(std::is_move_constructible<perf::sample>::value, "sample must be movable");
    static_assert(!std::is_copy_assignable<perf::sample>::value, "sample must not be copy-assignable");
    static_assert(!std::is_move_assignable<perf::sample>::value, "sample must not be move-assignable");
    SUCCEED();
}

TEST(NpuwPerfReport, IndentsByPathDepth) {
    ScopedProfile p("Model0/performance");
    p->handle("total@CPU")->add(100.0f);
    p->handle("total@CPU/submodel[000]")->add(40.0f);
    p->handle("total@CPU/submodel[001](fn)")->add(50.0f);

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();

    EXPECT_NE(npos, text.find("\n  total@CPU "));
    EXPECT_NE(npos, text.find("\n    submodel[000] "));
    EXPECT_NE(npos, text.find("\n    submodel[001](fn) "));
    // 100 - (40 + 50) = 10ms unaccounted for
    EXPECT_NE(npos, text.find("(unattributed)"));
}

TEST(NpuwPerfReport, ResidualWithinToleranceIsClamped) {
    ScopedProfile p("Model0/performance");
    p->handle("total@CPU")->add(100.0f);
    p->handle("total@CPU/submodel[000]")->add(60.0f);
    p->handle("total@CPU/submodel[001]")->add(40.0f);

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();

    EXPECT_EQ(npos, text.find("(unattributed)"));
    EXPECT_EQ(npos, text.find("(overlapped)"));
}

TEST(NpuwPerfReport, OverlappingChildrenNeverPrintANegativeResidual) {
    ScopedProfile p("Model0/performance");
    p->handle("total@CPU")->add(100.0f);
    p->handle("total@CPU/submodel[000]")->add(80.0f);
    p->handle("total@CPU/submodel[001]")->add(80.0f);

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();

    EXPECT_NE(npos, text.find("(overlapped)"));
    EXPECT_EQ(npos, text.find("-60"));
}

TEST(NpuwPerfReport, PyramidLevelIsAnAliasNotAPart) {
    ScopedProfile p("Model0/performance");
    const auto submodel = perf::tags::submodel("CPU", 1, true);
    p->handle(perf::tags::device("CPU"))->add(100.0f);
    p->handle(submodel)->add(100.0f);
    p->handle(perf::tags::pyramid(submodel, 2, 3072))->add(100.0f);

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();

    EXPECT_NE(npos, text.find("attn/pyramid[02] kv=3072"));
    EXPECT_NE(npos, text.find("(= parent)"));
    // The level _is_ the whole subgraph run, so no residual row is emitted for it
    EXPECT_EQ(npos, text.find("(unattributed)"));
}

TEST(NpuwPerfReport, SeveralPyramidLevelsPartitionTheParent) {
    // Chunked prefill can select different levels within one run - those rows are a
    // partition of the parent's calls, not aliases of it
    ScopedProfile p("Model0/performance");
    const auto submodel = perf::tags::submodel("CPU", 1, true);
    p->handle(perf::tags::device("CPU"))->add(100.0f);
    p->handle(submodel)->add(100.0f);
    p->handle(perf::tags::pyramid(submodel, 0, 1024))->add(40.0f);
    p->handle(perf::tags::pyramid(submodel, 1, 2048))->add(60.0f);

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();

    EXPECT_NE(npos, text.find("attn/pyramid[00] kv=1024"));
    EXPECT_NE(npos, text.find("attn/pyramid[01] kv=2048"));
    EXPECT_EQ(npos, text.find("(= parent)")) << "Levels that only cover part of the parent must not claim equality";
}

TEST(NpuwPerfReport, HfaFamilyGetsAnUnattributedResidual) {
    ScopedProfile p("Model0/performance");
    const auto submodel = perf::tags::submodel("CPU", 1, true);
    p->handle(perf::tags::device("CPU"))->add(100.0f);
    p->handle(submodel)->add(100.0f);
    p->handle(perf::tags::hfa_tile(submodel, 1024))->add(40.0f);
    p->handle(perf::tags::hfa_final_tile(submodel))->add(20.0f);
    p->handle(perf::tags::hfa_host_prep(submodel))->add(30.0f);

    std::ostringstream os;
    p->report(os);
    const auto text = os.str();

    EXPECT_NE(npos, text.find("attn/hfa/tile[1024]"));
    EXPECT_NE(npos, text.find("attn/hfa/tile[final]"));
    EXPECT_NE(npos, text.find("attn/hfa/host-prep"));
    EXPECT_NE(npos, text.find("(unattributed)"));
}

TEST(NpuwPerfReport, PrintsTheNestingLegend) {
    ScopedProfile p("Model0/performance");
    p->handle("total@CPU")->add(1.0f);

    std::ostringstream os;
    p->report(os);
    EXPECT_NE(npos, os.str().find("never sum across levels"));
}

TEST(NpuwPerfTags, PinTheNamingContract) {
    EXPECT_EQ("total@CPU", perf::tags::device("CPU"));
    EXPECT_EQ("total@NPU/submodel[003](fn)", perf::tags::submodel("NPU", 3, true));
    EXPECT_EQ("total@NPU/submodel[012]", perf::tags::submodel("NPU", 12, false));

    const auto submodel = perf::tags::submodel("NPU", 1, true);
    EXPECT_EQ(submodel + "/attn/pyramid[02] kv=3072", perf::tags::pyramid(submodel, 2, 3072));
    EXPECT_EQ(submodel + "/attn/hfa/tile[1024]", perf::tags::hfa_tile(submodel, 1024));
    EXPECT_EQ(submodel + "/attn/hfa/tile[final]", perf::tags::hfa_final_tile(submodel));
    EXPECT_EQ(submodel + "/attn/hfa/host-prep", perf::tags::hfa_host_prep(submodel));
}

TEST(NpuwPerfTags, SortLexicographicallyIntoTreeOrder) {
    // The '/'-separated paths must make std::map's ordering a valid pre-order walk
    const auto dev = perf::tags::device("CPU");
    const auto s0 = perf::tags::submodel("CPU", 0, false);
    const auto s1 = perf::tags::submodel("CPU", 1, true);
    const auto p1 = perf::tags::pyramid(s1, 0, 1024);

    EXPECT_LT(dev, s0);
    EXPECT_LT(s0, s1);
    EXPECT_LT(s1, p1);
}

#endif  // NPU_PLUGIN_DEVELOPER_BUILD
