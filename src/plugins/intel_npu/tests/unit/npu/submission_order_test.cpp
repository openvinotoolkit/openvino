// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "submission_order.hpp"

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"

using ::intel_npu::SubmissionOrder;
using ::intel_npu::SubmissionOrderPool;

namespace {

//
// Storing and reading back an event is not covered: Event requires an EventPool, which requires a
// Level Zero driver. Everything else here runs with no driver and no NPU, which is the point of
// having this state off IGraph.
//

/// The pool only ever uses a graph's address, so the bare interface is a sufficient stand-in.
/// is_profiling_blob is IGraph's only pure virtual; every other method keeps its throwing default,
/// and none of them is reachable from here.
class FakeGraph final : public intel_npu::IGraph {
public:
    std::optional<bool> is_profiling_blob() const override {
        return std::nullopt;
    }
};

TEST(SubmissionOrderTest, TicketsStartAtZeroAndIncrement) {
    SubmissionOrder order;

    EXPECT_EQ(order.next_id(), 0u);
    EXPECT_EQ(order.next_id(), 1u);
    EXPECT_EQ(order.next_id(), 2u);
}

TEST(SubmissionOrderTest, LastSubmittedIdStartsAtZero) {
    SubmissionOrder order;

    // The pipeline compares its ticket against this, so the initial value is load-bearing: the
    // first pipeline draws ticket 0 and is skipped by the ordering check.
    EXPECT_EQ(order.last_submitted_id(), 0u);
}

TEST(SubmissionOrderTest, LastSubmittedIdRoundTrips) {
    SubmissionOrder order;

    order.set_last_submitted_id(7);
    EXPECT_EQ(order.last_submitted_id(), 7u);

    order.set_last_submitted_id(8);
    EXPECT_EQ(order.last_submitted_id(), 8u);
}

TEST(SubmissionOrderTest, NoEventIsReportedBeforeAnythingIsSubmitted) {
    SubmissionOrder order;
    order.resize(2);

    EXPECT_EQ(order.last_event(0), nullptr);
    EXPECT_EQ(order.last_event(1), nullptr);
}

TEST(SubmissionOrderTest, QueryingAnUnsizedIndexReportsNoEventRatherThanReadingOutOfRange) {
    SubmissionOrder order;
    order.resize(1);

    // The previous implementation indexed the vector directly, so this was out-of-range access.
    // Reporting "nothing to wait on" is both safe and the answer the pipeline wants.
    EXPECT_EQ(order.last_event(1), nullptr);
    EXPECT_EQ(order.last_event(99), nullptr);
}

TEST(SubmissionOrderTest, QueryingBeforeAnyResizeReportsNoEvent) {
    SubmissionOrder order;

    EXPECT_EQ(order.last_event(0), nullptr);
}

TEST(SubmissionOrderTest, StoringAnEventOutsideTheSizedRangeIsRejected) {
    SubmissionOrder order;
    order.resize(1);

    // A null event is enough: the bounds check runs before the slot is touched.
    EXPECT_THROW(order.set_last_event(nullptr, 1), ov::Exception);
}

TEST(SubmissionOrderTest, ShrinkingDropsTheSlotsItRemoves) {
    SubmissionOrder order;
    order.resize(4);
    order.resize(1);

    // A pipeline with a smaller batch has nothing to wait on in the slots that went away.
    EXPECT_EQ(order.last_event(1), nullptr);
    EXPECT_EQ(order.last_event(3), nullptr);
}

TEST(SubmissionOrderPoolTest, SiblingsOfOneGraphShareOneInstance) {
    FakeGraph graph;
    auto& pool = SubmissionOrderPool::getInstance();

    const auto first = pool.get(graph);
    const auto second = pool.get(graph);

    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first, second);

    // The shared instance is what makes the tickets a single sequence per graph.
    EXPECT_EQ(first->next_id(), 0u);
    EXPECT_EQ(second->next_id(), 1u);
}

TEST(SubmissionOrderPoolTest, DifferentGraphsAreOrderedIndependently) {
    FakeGraph first_graph;
    FakeGraph second_graph;
    auto& pool = SubmissionOrderPool::getInstance();

    const auto first = pool.get(first_graph);
    const auto second = pool.get(second_graph);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first, second);

    EXPECT_EQ(first->next_id(), 0u);
    EXPECT_EQ(second->next_id(), 0u);
}

TEST(SubmissionOrderPoolTest, StateOutlivesIndividualHoldersButNotAllOfThem) {
    FakeGraph graph;
    auto& pool = SubmissionOrderPool::getInstance();

    auto held = pool.get(graph);
    const auto* const original = held.get();
    EXPECT_EQ(pool.get(graph).get(), original);  // dropping a temporary holder keeps it alive

    held->set_last_submitted_id(5);
    held.reset();

    // With no pipeline left the ordering constraint is vacuous, so the next one starts over. This
    // is also what makes the pool's raw-pointer keys safe: a pipeline holds a shared_ptr to its
    // graph, so a graph can only be destroyed once its state has expired exactly like this, and an
    // address that gets recycled lands on a dead entry rather than on another graph's state.
    const auto fresh = pool.get(graph);
    EXPECT_EQ(fresh->next_id(), 0u);
    EXPECT_EQ(fresh->last_submitted_id(), 0u);
}

}  // namespace
