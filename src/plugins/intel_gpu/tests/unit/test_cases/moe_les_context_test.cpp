#include "intel_gpu/runtime/les_context.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <vector>

using namespace std::chrono_literals;

namespace {

TEST(moe_les_lifecycle, rejects_invalid_capacity) {
    EXPECT_THROW(cldnn::les_context(0), std::invalid_argument);
    EXPECT_THROW(cldnn::les_context(1025), std::invalid_argument);
}

TEST(moe_les_lifecycle, one_worker_and_fifo_completion) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    std::vector<size_t> order;
    std::set<std::thread::id> workers;
    std::vector<cldnn::les_context::ticket> tickets;
    for (size_t index = 0; index < 32; ++index) {
        tickets.push_back(context.submit(epoch, [&, index] {
            order.push_back(index);
            workers.insert(std::this_thread::get_id());
        }));
    }
    context.wait();
    for (size_t index = 0; index < tickets.size(); ++index) {
        EXPECT_EQ(tickets[index].epoch, epoch);
        EXPECT_NO_THROW(tickets[index].completion.get());
        EXPECT_EQ(order[index], index);
    }
    EXPECT_EQ(workers.size(), 1U);
    EXPECT_EQ(workers.count(std::this_thread::get_id()), 0U);
}

TEST(moe_les_lifecycle, bounded_queue_blocks_producer_until_space_is_available) {
    cldnn::les_context context(1);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto first = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    auto second = context.submit(epoch, [] {});
    std::promise<void> submitting;
    auto producer = std::async(std::launch::async, [&] {
        submitting.set_value();
        return context.submit(epoch, [] {});
    });
    EXPECT_EQ(submitting.get_future().wait_for(5s), std::future_status::ready);
    EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
    release.set_value();
    EXPECT_EQ(producer.wait_for(5s), std::future_status::ready);
    EXPECT_NO_THROW(producer.get().completion.get());
    EXPECT_NO_THROW(first.completion.get());
    EXPECT_NO_THROW(second.completion.get());
}

TEST(moe_les_lifecycle, cancellation_terminates_queued_and_active_tickets) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto first = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    std::atomic<bool> ran{false};
    auto second = context.submit(epoch, [&] { ran = true; });
    auto cancelling = std::async(std::launch::async, [&] { context.cancel(); });
    EXPECT_EQ(second.completion.wait_for(5s), std::future_status::ready);
    EXPECT_THROW(second.completion.get(), cldnn::les_task_cancelled);
    EXPECT_FALSE(ran);
    EXPECT_EQ(cancelling.wait_for(20ms), std::future_status::timeout);
    release.set_value();
    cancelling.get();
    EXPECT_THROW(first.completion.get(), cldnn::les_task_cancelled);
    EXPECT_THROW(context.submit(epoch, [] {}), cldnn::les_task_cancelled);
}

TEST(moe_les_lifecycle, worker_error_wakes_all_tickets_and_next_epoch_recovers) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto first = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
        throw std::runtime_error("fake weight read failed");
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    std::atomic<bool> ran{false};
    auto second = context.submit(epoch, [&] { ran = true; });
    release.set_value();
    EXPECT_EQ(first.completion.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(second.completion.wait_for(5s), std::future_status::ready);
    EXPECT_THROW(first.completion.get(), std::runtime_error);
    EXPECT_THROW(second.completion.get(), std::runtime_error);
    EXPECT_THROW(context.wait(), std::runtime_error);
    EXPECT_THROW(context.submit(epoch, [] {}), std::runtime_error);
    EXPECT_FALSE(ran);
    auto next = context.begin_epoch();
    EXPECT_GT(next, epoch);
    EXPECT_NO_THROW(context.submit(next, [] {}).completion.get());
}

TEST(moe_les_lifecycle, epoch_rotation_waits_for_active_work_and_discards_old_queue) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto first = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    auto second = context.submit(epoch, [] {});
    auto rotating = std::async(std::launch::async, [&] { return context.begin_epoch(); });
    EXPECT_EQ(second.completion.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(rotating.wait_for(20ms), std::future_status::timeout);
    release.set_value();
    auto next = rotating.get();
    EXPECT_GT(next, epoch);
    EXPECT_THROW(first.completion.get(), cldnn::les_task_cancelled);
    EXPECT_THROW(second.completion.get(), cldnn::les_task_cancelled);
    EXPECT_NO_THROW(context.submit(next, [] {}).completion.get());
}

TEST(moe_les_lifecycle, stopping_wakes_a_blocked_producer_and_is_idempotent) {
    cldnn::les_context context(1);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto first = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    auto queued = context.submit(epoch, [] {});
    auto producer = std::async(std::launch::async, [&] { return context.submit(epoch, [] {}); });
    auto stopping = std::async(std::launch::async, [&] { context.stop(); });
    EXPECT_EQ(producer.wait_for(5s), std::future_status::ready);
    EXPECT_THROW(producer.get(), cldnn::les_task_cancelled);
    EXPECT_EQ(queued.completion.wait_for(5s), std::future_status::ready);
    EXPECT_THROW(queued.completion.get(), cldnn::les_task_cancelled);
    release.set_value();
    stopping.get();
    EXPECT_THROW(first.completion.get(), cldnn::les_task_cancelled);
    EXPECT_NO_THROW(context.stop());
    EXPECT_THROW(context.begin_epoch(), cldnn::les_task_cancelled);
}

TEST(moe_les_lifecycle, worker_reentrant_control_is_reported_as_task_failure) {
    cldnn::les_context context(1);
    auto epoch = context.begin_epoch();
    auto work = context.submit(epoch, [&] { context.wait(); });
    EXPECT_THROW(work.completion.get(), std::logic_error);
    EXPECT_THROW(context.wait(), std::logic_error);
}

TEST(moe_les_lifecycle, independent_contexts_and_repeated_destruction) {
    for (size_t iteration = 0; iteration < 16; ++iteration) {
        cldnn::les_context first(1);
        cldnn::les_context second(1);
        auto first_epoch = first.begin_epoch();
        auto second_epoch = second.begin_epoch();
        first.cancel();
        EXPECT_THROW(first.submit(first_epoch, [] {}), cldnn::les_task_cancelled);
        EXPECT_NO_THROW(second.submit(second_epoch, [] {}).completion.get());
    }
}

TEST(moe_les_lifecycle, rejects_empty_task) {
    cldnn::les_context context(1);
    EXPECT_THROW(context.submit(context.begin_epoch(), {}), std::invalid_argument);
}

TEST(moe_les_scheduler, demand_preempts_pending_speculation) {
    cldnn::les_context context(4);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    std::vector<int> order;
    auto speculative = context.try_submit_speculative(epoch, [&] { order.push_back(3); });
    auto second_speculative = context.try_submit_speculative(epoch, [&] { order.push_back(4); });
    auto demand = context.submit(epoch, [&] { order.push_back(1); });
    auto second_demand = context.submit(epoch, [&] { order.push_back(2); });
    release.set_value();
    context.wait();
    ASSERT_TRUE(speculative);
    ASSERT_TRUE(second_speculative);
    EXPECT_NO_THROW(active.completion.get());
    EXPECT_NO_THROW(demand.completion.get());
    EXPECT_NO_THROW(second_demand.completion.get());
    EXPECT_NO_THROW(speculative->completion.get());
    EXPECT_NO_THROW(second_speculative->completion.get());
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3, 4}));
}

TEST(moe_les_scheduler, full_speculative_queue_yields_to_demand) {
    cldnn::les_context context(1);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    std::atomic<bool> ran{false};
    auto speculative = context.try_submit_speculative(epoch, [&] { ran = true; });
    auto rejected = context.try_submit_speculative(epoch, [] {});
    auto producer = std::async(std::launch::async, [&] { return context.submit(epoch, [] {}); });
    const auto status = producer.wait_for(5s);
    release.set_value();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_FALSE(rejected);
    auto demand = producer.get();
    context.wait();
    ASSERT_TRUE(speculative);
    EXPECT_THROW(speculative->completion.get(), cldnn::les_task_cancelled);
    EXPECT_NO_THROW(demand.completion.get());
    EXPECT_NO_THROW(active.completion.get());
    EXPECT_FALSE(ran);
}

TEST(moe_les_scheduler, speculative_failure_does_not_poison_demand) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    auto speculative = context.try_submit_speculative(epoch, [] { throw std::runtime_error("speculation failed"); });
    ASSERT_TRUE(speculative);
    EXPECT_THROW(speculative->completion.get(), std::runtime_error);
    EXPECT_NO_THROW(context.submit(epoch, [] {}).completion.get());
    EXPECT_NO_THROW(context.wait());
}

TEST(moe_les_scheduler, pending_prefetch_promoted_to_demand_without_duplicate_work) {
    cldnn::les_context context(3);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    std::vector<int> order;
    auto first = context.try_submit_speculative(epoch, [&] { order.push_back(2); });
    auto matched = context.try_submit_speculative(epoch, [&] { order.push_back(1); });
    EXPECT_TRUE(context.promote(*matched));
    EXPECT_FALSE(context.promote(*matched));
    EXPECT_FALSE(context.cancel_pending(*matched));
    release.set_value();
    context.wait();
    EXPECT_NO_THROW(active.completion.get());
    EXPECT_NO_THROW(first->completion.get());
    EXPECT_NO_THROW(matched->completion.get());
    EXPECT_EQ(order, (std::vector<int>{1, 2}));
}

TEST(moe_les_scheduler, pending_cancel_does_not_cancel_active_upload) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.try_submit_speculative(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    auto queued = context.try_submit_speculative(epoch, [] {});
    EXPECT_FALSE(context.cancel_pending(*active));
    EXPECT_TRUE(context.cancel_pending(*queued));
    EXPECT_FALSE(context.cancel_pending(*queued));
    release.set_value();
    EXPECT_NO_THROW(active->completion.get());
    EXPECT_THROW(queued->completion.get(), cldnn::les_task_cancelled);
    EXPECT_NO_THROW(context.wait());
}

TEST(moe_les_scheduler, epoch_rotation_discards_pending_speculation) {
    cldnn::les_context context(2);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    std::atomic<bool> ran{false};
    auto speculative = context.try_submit_speculative(epoch, [&] { ran = true; });
    auto rotating = std::async(std::launch::async, [&] { return context.begin_epoch(); });
    const auto status = speculative->completion.wait_for(5s);
    release.set_value();
    auto next = rotating.get();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_THROW(speculative->completion.get(), cldnn::les_task_cancelled);
    EXPECT_THROW(active.completion.get(), cldnn::les_task_cancelled);
    EXPECT_FALSE(ran);
    EXPECT_THROW(context.try_submit_speculative(epoch, [] {}), cldnn::les_task_cancelled);
    EXPECT_NO_THROW(context.try_submit_speculative(next, [] {})->completion.get());
}

TEST(moe_les_lifecycle, destructor_joins_active_work_and_cancels_pending_work) {
    auto context = std::make_unique<cldnn::les_context>(1);
    auto epoch = context->begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto first = context->submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    auto pending = context->submit(epoch, [] {});
    auto destroying = std::async(std::launch::async, [&] { context.reset(); });
    EXPECT_EQ(pending.completion.wait_for(5s), std::future_status::ready);
    EXPECT_THROW(pending.completion.get(), cldnn::les_task_cancelled);
    EXPECT_EQ(destroying.wait_for(20ms), std::future_status::timeout);
    release.set_value();
    destroying.get();
    EXPECT_THROW(first.completion.get(), cldnn::les_task_cancelled);
}

TEST(moe_les_lifecycle, cancel_releases_unstarted_task_resources) {
    cldnn::les_context context(1);
    auto epoch = context.begin_epoch();
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(5s), std::future_status::ready);
    auto resource = std::make_shared<int>(42);
    std::weak_ptr<int> observer = resource;
    auto pending = context.submit(epoch, [resource] {});
    resource.reset();
    auto cancelling = std::async(std::launch::async, [&] { context.cancel(); });
    EXPECT_EQ(pending.completion.wait_for(5s), std::future_status::ready);
    release.set_value();
    cancelling.get();
    EXPECT_TRUE(observer.expired());
    EXPECT_THROW(pending.completion.get(), cldnn::les_task_cancelled);
    EXPECT_THROW(active.completion.get(), cldnn::les_task_cancelled);
}

}  // namespace