// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "submission_order.hpp"

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"

using ::intel_npu::SubmissionOrder;

namespace {

//
// Storing and reading back an event is not covered: Event requires an EventPool, which requires a
// Level Zero driver. Everything else here runs with no driver and no NPU, which is the point of
// having this state off IGraph.
//

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

}  // namespace
