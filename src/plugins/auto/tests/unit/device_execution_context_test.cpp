// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_execution_context.hpp"
#include "instance_registry.hpp"

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
        return context.pool().idle_count();
    }
};

TEST_F(DeviceExecutionContextTest, startsUnloaded) {
    DeviceExecutionContext context("GPU.0");
    EXPECT_EQ(context.name(), "GPU.0");
    EXPECT_EQ(context.state(), State::Unloaded);
    EXPECT_FALSE(context.is_ready());
    EXPECT_FALSE(context.is_kept_alive());
    EXPECT_TRUE(context.last_error().empty());
    EXPECT_EQ(context.pool().size(), 0u);
    EXPECT_EQ(count_idle(context), 0u);
}

TEST_F(DeviceExecutionContextTest, makeReadyCreatesWorkersAndSetsUpEachOne) {
    DeviceExecutionContext context("GPU.0");
    EXPECT_EQ(make_ready(context, 3), 3u);
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_TRUE(context.compiled_model()._ptr);
    ASSERT_EQ(context.pool().size(), 3u);
    for (size_t i = 0; i < context.pool().size(); i++) {
        EXPECT_EQ(context.pool().worker(i).m_index, static_cast<int>(i));
    }
    EXPECT_EQ(count_idle(context), 3u);
}

TEST_F(DeviceExecutionContextTest, makeReadyWhileReadyThrows) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    EXPECT_THROW(make_ready(context), ov::Exception);
    EXPECT_EQ(context.pool().size(), 2u);
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
    EXPECT_EQ(context.pool().size(), 0u);
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
    EXPECT_EQ(context.pool().size(), 0u);
    EXPECT_FALSE(context.compiled_model()._ptr);
    EXPECT_FALSE(context.try_release());
}

TEST_F(DeviceExecutionContextTest, releaseIsRefusedWhileATaskIsQueued) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context);
    context.pool().enqueue([] {});
    EXPECT_FALSE(context.try_release());
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_EQ(count_idle(context), 2u);
    ov::threading::Task task;
    ASSERT_TRUE(context.pool().try_dequeue(task));
    EXPECT_TRUE(context.try_release());
}

TEST_F(DeviceExecutionContextTest, modelCopyTakenBeforeReleaseStaysValid) {
    auto tracked = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, mock_plugin_gpu);
    const std::weak_ptr<ov::MockICompiledModel> weak_model = tracked;
    ON_CALL(*tracked, inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*tracked, outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    ON_CALL(*tracked, create_sync_infer_request()).WillByDefault([weak_model]() {
        return std::make_shared<MockISyncInferRequest>(weak_model.lock());
    });
    ON_CALL(*tracked, create_infer_request()).WillByDefault([weak_model]() {
        return weak_model.lock()->ICompiledModel::create_infer_request();
    });
    DeviceExecutionContext context("GPU.0");
    context.make_ready(ov::auto_plugin::SoCompiledModel{tracked, {}}, 2, [](DeviceExecutionContext&, WorkerInferRequest&) {});
    tracked.reset();
    // the copy outlives the release, so the model must not be destroyed by it
    auto query_copy = context.compiled_model();
    EXPECT_TRUE(context.try_release());
    EXPECT_FALSE(context.compiled_model());
    EXPECT_FALSE(weak_model.expired());
    query_copy = ov::auto_plugin::SoCompiledModel{};
    EXPECT_TRUE(weak_model.expired());
}

TEST_F(DeviceExecutionContextTest, releasedContextRejectsLateIdleWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    WorkerInferRequest* worker = &context.pool().worker(0);
    ASSERT_TRUE(context.try_release());
    EXPECT_FALSE(context.pool().adopt(worker).give_back());
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
    auto busy = context.pool().try_acquire();
    ASSERT_TRUE(busy);
    EXPECT_FALSE(context.try_release());
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_EQ(context.pool().size(), 2u);
    EXPECT_EQ(count_idle(context), 1u);
    // the busy worker comes back and the device can be released
    ASSERT_TRUE(busy.give_back());
    EXPECT_EQ(count_idle(context), 2u);
    EXPECT_TRUE(context.try_release());
}

TEST_F(DeviceExecutionContextTest, makeReadyAfterReleaseRestoresWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 2);
    ASSERT_TRUE(context.try_release());
    EXPECT_EQ(make_ready(context, 1), 1u);
    EXPECT_EQ(context.state(), State::Ready);
    EXPECT_EQ(context.pool().size(), 1u);
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
    auto busy = context.pool().try_acquire();
    ASSERT_TRUE(busy);
    context.stop_scheduling();
    EXPECT_FALSE(busy.give_back());
    EXPECT_FALSE(context.pool().try_acquire());
    EXPECT_EQ(context.state(), State::Ready);
}

TEST_F(DeviceExecutionContextTest, releaseKeepsInferTimesForTheSummary) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    const auto start = std::chrono::steady_clock::now();
    const auto end = start + std::chrono::milliseconds(5);
    context.pool().worker(0).m_start_times.push_back(start);
    context.pool().worker(0).m_end_times.push_back(end);
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
    context.pool().enqueue([&ran] {
        ran = true;
    });
    EXPECT_TRUE(context.pool().has_queued_tasks());
    ov::threading::Task task;
    ASSERT_TRUE(context.pool().try_dequeue(task));
    task();
    EXPECT_TRUE(ran);
    EXPECT_FALSE(context.pool().try_dequeue(task));
}

TEST_F(DeviceExecutionContextTest, logSummaryIncludesReleasedWorkers) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    const auto start = std::chrono::steady_clock::now();
    context.pool().worker(0).m_start_times.push_back(start);
    context.pool().worker(0).m_end_times.push_back(start + std::chrono::milliseconds(5));
    ASSERT_TRUE(context.try_release());
    EXPECT_NO_THROW(context.log_summary("AUTO", start - std::chrono::seconds(1), true));
}

TEST_F(DeviceExecutionContextTest, startupHelperIsADifferentInstanceOfTheSameDevice) {
    const ov::auto_plugin::ModelInstanceKey main{"CPU"};
    const ov::auto_plugin::ModelInstanceKey helper{"CPU", ov::auto_plugin::ModelInstanceKey::Profile::StartupHelper};
    EXPECT_NE(main, helper);
    EXPECT_TRUE(main < helper || helper < main);
    EXPECT_TRUE(helper.is_startup_helper());
    EXPECT_FALSE(main.is_startup_helper());
    DeviceExecutionContext context(helper);
    EXPECT_EQ(context.name(), "CPU");
    EXPECT_EQ(context.key(), helper);
}

TEST_F(DeviceExecutionContextTest, leaseGivesTheWorkerBackWhenDestroyed) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 2);
    {
        auto lease = context.pool().try_acquire();
        ASSERT_TRUE(lease);
        EXPECT_EQ(lease.get()->m_index, 0);
        EXPECT_EQ(context.pool().leased_count(), 1u);
        EXPECT_EQ(count_idle(context), 1u);
    }
    EXPECT_EQ(context.pool().leased_count(), 0u);
    EXPECT_EQ(count_idle(context), 2u);
}

TEST_F(DeviceExecutionContextTest, detachedLeaseKeepsTheWorkerUntilItIsAdopted) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 2);
    WorkerInferRequest* in_flight = nullptr;
    {
        auto lease = context.pool().try_acquire();
        in_flight = lease.detach();
    }
    ASSERT_NE(in_flight, nullptr);
    EXPECT_EQ(context.pool().leased_count(), 1u);
    EXPECT_FALSE(context.try_release());
    // the next lease gets the other worker
    {
        auto other = context.pool().try_acquire();
        ASSERT_TRUE(other);
        EXPECT_NE(other.get(), in_flight);
    }
    EXPECT_TRUE(context.pool().adopt(in_flight).give_back());
    EXPECT_EQ(context.pool().leased_count(), 0u);
    EXPECT_TRUE(context.try_release());
}

TEST_F(DeviceExecutionContextTest, noWorkerIsLeasedFromAnExhaustedPool) {
    DeviceExecutionContext context("GPU.0");
    make_ready(context, 1);
    auto first = context.pool().try_acquire();
    ASSERT_TRUE(first);
    EXPECT_FALSE(context.pool().try_acquire());
}

TEST_F(DeviceExecutionContextTest, registryAnswersWithThePublishedInstance) {
    ov::auto_plugin::InstanceRegistry registry;
    auto& gpu = registry.add("GPU.0");
    auto& cpu = registry.add("CPU");
    EXPECT_EQ(&registry.add("GPU.0"), &gpu);
    EXPECT_FALSE(registry.active_device().compiled_model);
    make_ready(gpu);
    make_ready(cpu);
    registry.publish(ov::auto_plugin::ModelInstanceKey("GPU.0"));
    auto active = registry.active_device();
    EXPECT_EQ(active.instance, ov::auto_plugin::ModelInstanceKey("GPU.0"));
    EXPECT_TRUE(active.compiled_model);
    registry.publish([]() -> std::optional<ov::auto_plugin::ModelInstanceKey> {
        return ov::auto_plugin::ModelInstanceKey("CPU");
    });
    EXPECT_EQ(registry.active_device().instance, ov::auto_plugin::ModelInstanceKey("CPU"));
    // the previous instance is released once the other one is published
    EXPECT_TRUE(gpu.try_release());
    EXPECT_TRUE(registry.active_device().compiled_model);
}

TEST_F(DeviceExecutionContextTest, registryReportsNothingForAReleasedInstance) {
    ov::auto_plugin::InstanceRegistry registry;
    auto& gpu = registry.add("GPU.0");
    make_ready(gpu);
    registry.publish(ov::auto_plugin::ModelInstanceKey("GPU.0"));
    ASSERT_TRUE(gpu.try_release());
    EXPECT_FALSE(registry.active_device().compiled_model);
    registry.clear();
    EXPECT_FALSE(registry.active_instance());
    EXPECT_EQ(registry.find("GPU.0"), nullptr);
}
