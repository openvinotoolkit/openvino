// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_execution_context.hpp"

#include "include/auto_unit_test.hpp"

using namespace ov::mock_auto_plugin;
using ov::auto_plugin::DeviceExecutionContext;
using State = DeviceExecutionContext::State;

class DeviceExecutionContextTest : public tests::BaseTest, public ::testing::Test {
public:
    ov::auto_plugin::SoCompiledModel compiled_model() const {
        return {mockIExeNetActual, {}};
    }

    size_t make_ready(DeviceExecutionContext& context, size_t num_requests = 2) {
        size_t set_up = 0;
        context.make_ready(compiled_model(), num_requests, [&set_up](DeviceExecutionContext&, WorkerInferRequest&) {
            set_up++;
        });
        return set_up;
    }

    static size_t count_idle(DeviceExecutionContext& context) {
        size_t count = 0;
        std::pair<int, WorkerInferRequest*> idle;
        std::vector<std::pair<int, WorkerInferRequest*>> popped;
        while (context.idle_workers().try_pop(idle)) {
            popped.push_back(idle);
            count++;
        }
        for (auto& worker : popped) {
            context.idle_workers().try_push(std::move(worker));
        }
        return count;
    }
};

TEST_F(DeviceExecutionContextTest, startsUnloaded) {
    DeviceExecutionContext context("GPU.0");
    EXPECT_EQ(context.name(), "GPU.0");
    EXPECT_EQ(context.state(), State::Unloaded);
    EXPECT_FALSE(context.is_ready());
    EXPECT_FALSE(context.is_kept_alive());
    EXPECT_TRUE(context.last_error().empty());
    EXPECT_TRUE(context.workers().empty());
    EXPECT_EQ(count_idle(context), 0u);
}

TEST_F(DeviceExecutionContextTest, makeReadyCreatesWorkersAndSetsUpEachOne) {
    DeviceExecutionContext context("GPU.0");
    EXPECT_EQ(make_ready(context, 3), 3u);
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_TRUE(context.compiled_model()._ptr);
    ASSERT_EQ(context.workers().size(), 3u);
    for (size_t i = 0; i < context.workers().size(); i++) {
        EXPECT_EQ(context.workers()[i].m_index, static_cast<int>(i));
    }
    EXPECT_EQ(count_idle(context), 3u);
}

TEST_F(DeviceExecutionContextTest, makeReadyWhileReadyThrows) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    EXPECT_THROW(make_ready(context), ov::Exception);
    EXPECT_EQ(context.workers().size(), 2u);
    EXPECT_EQ(count_idle(context), 2u);
}

TEST_F(DeviceExecutionContextTest, failedMakeReadyLeavesContextUnloaded) {
    DeviceExecutionContext context("GPU.0");
    EXPECT_THROW(context.make_ready(compiled_model(),
                                    2,
                                    [](DeviceExecutionContext&, WorkerInferRequest&) {
                                        OPENVINO_THROW("worker setup failure");
                                    }),
                 ov::Exception);
    EXPECT_EQ(context.state(), State::Unloaded);
    EXPECT_TRUE(context.workers().empty());
    EXPECT_FALSE(context.compiled_model()._ptr);
    EXPECT_EQ(count_idle(context), 0u);
    EXPECT_NO_THROW(make_ready(context));
    EXPECT_EQ(context.state(), State::Ready);
}

TEST_F(DeviceExecutionContextTest, releaseDropsWorkersAndModel) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    EXPECT_TRUE(context.try_release());
    EXPECT_EQ(context.state(), State::Released);
    EXPECT_TRUE(context.workers().empty());
    EXPECT_FALSE(context.compiled_model()._ptr);
    EXPECT_FALSE(context.try_release());
}

TEST_F(DeviceExecutionContextTest, releasedContextRejectsLateIdleWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    WorkerInferRequest* worker = &context.workers().front();
    ASSERT_TRUE(context.try_release());
    EXPECT_FALSE(context.idle_workers().try_push(std::make_pair(0, worker)));
}

TEST_F(DeviceExecutionContextTest, contextKeptAliveIsNeverReleased) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    context.keep_alive();
    EXPECT_TRUE(context.is_kept_alive());
    EXPECT_FALSE(context.try_release());
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_EQ(count_idle(context), 2u);
}

TEST_F(DeviceExecutionContextTest, busyWorkerBlocksReleaseWithoutSideEffects) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 2);
    std::pair<int, WorkerInferRequest*> busy;
    ASSERT_TRUE(context.idle_workers().try_pop(busy));
    EXPECT_FALSE(context.try_release());
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_EQ(context.workers().size(), 2u);
    EXPECT_EQ(count_idle(context), 1u);
    // the busy worker comes back and the device can be released
    ASSERT_TRUE(context.idle_workers().try_push(std::move(busy)));
    EXPECT_EQ(count_idle(context), 2u);
    EXPECT_TRUE(context.try_release());
}

TEST_F(DeviceExecutionContextTest, makeReadyAfterReleaseRestoresWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 2);
    ASSERT_TRUE(context.try_release());
    EXPECT_EQ(make_ready(context, 1), 1u);
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_EQ(context.workers().size(), 1u);
    EXPECT_EQ(count_idle(context), 1u);
}

TEST_F(DeviceExecutionContextTest, faultedKeepsTheLatestCompileError) {
    DeviceExecutionContext context("GPU.0");
    context.mark_faulted("first failure");
    EXPECT_EQ(context.state(), State::Faulted);
    EXPECT_EQ(context.last_error(), "first failure");
    context.mark_faulted("second failure");
    EXPECT_EQ(context.state(), State::Faulted);
    EXPECT_EQ(context.last_error(), "second failure");
    EXPECT_FALSE(context.try_release());
    // a later attempt may still succeed
    make_ready(context);
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_TRUE(context.last_error().empty());
}

TEST_F(DeviceExecutionContextTest, readyContextIgnoresCompileFailure) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    context.mark_faulted("failure");
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_TRUE(context.last_error().empty());
}

TEST_F(DeviceExecutionContextTest, releasedContextCanBecomeFaulted) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    ASSERT_TRUE(context.try_release());
    context.mark_faulted("reload failure");
    EXPECT_EQ(context.state(), State::Faulted);
    EXPECT_EQ(context.last_error(), "reload failure");
}

TEST_F(DeviceExecutionContextTest, stopSchedulingRejectsIdleWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    std::pair<int, WorkerInferRequest*> busy;
    ASSERT_TRUE(context.idle_workers().try_pop(busy));
    context.stop_scheduling();
    EXPECT_FALSE(context.idle_workers().try_push(std::move(busy)));
    EXPECT_EQ(context.state(), State::Ready);
}

TEST_F(DeviceExecutionContextTest, releaseKeepsInferTimesForTheSummary) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    const auto start = std::chrono::steady_clock::now();
    const auto end = start + std::chrono::milliseconds(5);
    context.workers().front().m_start_times.push_back(start);
    context.workers().front().m_end_times.push_back(end);
    ASSERT_TRUE(context.try_release());
    auto [starts, ends] = context.take_released_stats();
    ASSERT_EQ(starts.size(), 1u);
    ASSERT_EQ(ends.size(), 1u);
    EXPECT_EQ(starts.front(), start);
    EXPECT_EQ(ends.front(), end);
    // handed over once
    auto [again_starts, again_ends] = context.take_released_stats();
    EXPECT_TRUE(again_starts.empty());
    EXPECT_TRUE(again_ends.empty());
}

TEST_F(DeviceExecutionContextTest, pendingTasksAreOwnedByTheContext) {
    DeviceExecutionContext context("GPU.0");
    bool ran = false;
    context.pending_tasks().push([&ran] {
        ran = true;
    });
    ov::threading::Task task;
    ASSERT_TRUE(context.pending_tasks().try_pop(task));
    task();
    EXPECT_TRUE(ran);
    EXPECT_FALSE(context.pending_tasks().try_pop(task));
}

TEST_F(DeviceExecutionContextTest, cacheInfoIsKeptPerContext) {
    DeviceExecutionContext context("GPU.0");
    EXPECT_FALSE(context.cache_info().supports_caching.has_value());
    context.cache_info().supports_caching = true;
    context.cache_info().blob_id = "blob";
    make_ready(context);
    ASSERT_TRUE(context.try_release());
    EXPECT_EQ(context.cache_info().supports_caching, std::optional<bool>(true));
    EXPECT_EQ(context.cache_info().blob_id, std::optional<std::string>("blob"));
}

TEST_F(DeviceExecutionContextTest, logSummaryIncludesReleasedWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    const auto start = std::chrono::steady_clock::now();
    context.workers().front().m_start_times.push_back(start);
    context.workers().front().m_end_times.push_back(start + std::chrono::milliseconds(5));
    ASSERT_TRUE(context.try_release());
    EXPECT_NO_THROW(context.log_summary("AUTO", start - std::chrono::seconds(1), true));
}
