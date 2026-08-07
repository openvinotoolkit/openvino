// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_continuation.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "llm_stored_tokens_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {

using ov::npuw::ContinuationCoordinator;
using Stage = ov::npuw::ContinuationCoordinator::Stage;

constexpr uint32_t CHUNK = 1024u;
constexpr uint32_t MAX_PROMPT = 4096u;

ContinuationCoordinator make_enabled() {
    ContinuationCoordinator c;
    c.enable(CHUNK, MAX_PROMPT);
    return c;
}

// Brings the coordinator to an idle state with the given committed prefill history.
ContinuationCoordinator make_after_prefill(uint32_t prefill_tokens, uint32_t generated_tokens = 0u) {
    auto c = make_enabled();
    c.commit_prefill(prefill_tokens);
    if (generated_tokens > 0u) {
        c.publish_generate(prefill_tokens + generated_tokens);
    }
    return c;
}

ov::SoPtr<ov::ITensor> scalar_i64(int64_t value) {
    auto tensor = ov::Tensor(ov::element::i64, ov::Shape{1});
    tensor.data<int64_t>()[0] = value;
    return ov::get_tensor_impl(tensor);
}

int64_t read_scalar(const ov::SoPtr<ov::ITensor>& tensor) {
    return tensor->data<int64_t>()[0];
}

// The keep rule. K = C * floor(min(K_common, K_max, W) / C).

TEST(NPUWContinuation, KeepRuleGrantsAlignedValue) {
    auto c = make_after_prefill(3000u, 50u);
    // W = 3000, K_max = 3072, propose the full live history.
    EXPECT_EQ(c.propose(3050), 2048u);
    EXPECT_EQ(c.stage(), Stage::PENDING);
}

TEST(NPUWContinuation, KeepRuleClampsToWatermark) {
    auto c = make_after_prefill(2048u, 500u);
    // 2548 live tokens but only 2048 were produced by prefill.
    EXPECT_EQ(c.propose(2548), 2048u);
}

TEST(NPUWContinuation, KeepRuleClampsToCapacity) {
    ContinuationCoordinator c;
    c.enable(1024u, 4096u);
    c.commit_prefill(4000u);
    // K_max = 4096 - 1024 = 3072, rounded stays 3072.
    EXPECT_EQ(c.propose(4000), 3072u);
}

TEST(NPUWContinuation, KeepRuleRoundsDownToChunk) {
    auto c = make_after_prefill(1500u);
    EXPECT_EQ(c.propose(1500), 1024u);
}

TEST(NPUWContinuation, SubChunkHistoryGrantsZeroAndArmsReset) {
    auto c = make_after_prefill(1000u);
    EXPECT_EQ(c.propose(1000), 0u);
    // A zero keep is a full reset, not a continuation with an empty prefix.
    EXPECT_EQ(c.stage(), Stage::PENDING);
    EXPECT_EQ(c.pending(), 0u);
    EXPECT_EQ(c.query(), 0);
}

TEST(NPUWContinuation, GrowthIsRejected) {
    auto c = make_after_prefill(2048u);
    EXPECT_THROW(c.propose(2049), ov::Exception);
    // A rejected proposal leaves the request idle with the committed count readable.
    EXPECT_EQ(c.stage(), Stage::IDLE);
    EXPECT_EQ(c.query(), 2048);
}

TEST(NPUWContinuation, NegativeProposalIsRejected) {
    auto c = make_after_prefill(2048u);
    EXPECT_THROW(c.propose(-1), ov::Exception);
}

TEST(NPUWContinuation, SecondProposalWhilePendingIsRejected) {
    auto c = make_after_prefill(3072u);
    EXPECT_EQ(c.propose(3072), 3072u);
    EXPECT_THROW(c.propose(2048), ov::Exception);
}

TEST(NPUWContinuation, ProposalWhileResetPendingIsRejected) {
    auto c = make_after_prefill(2048u);
    c.request_reset();
    EXPECT_THROW(c.propose(2048), ov::Exception);
}

// The grant readback.

TEST(NPUWContinuation, QueryReturnsLiveTokensWhileIdle) {
    auto c = make_after_prefill(2048u, 10u);
    EXPECT_EQ(c.query(), 2058);
}

TEST(NPUWContinuation, QueryReturnsGrantWhilePending) {
    auto c = make_after_prefill(3000u);
    c.propose(3000);
    EXPECT_EQ(c.query(), 2048);
}

TEST(NPUWContinuation, PublishingLiveCountDoesNotReArmCommand) {
    auto c = make_after_prefill(2048u);
    c.propose(2048);
    c.commit_prefill(2500u);
    // After the commit there is no pending command, only the published count.
    EXPECT_EQ(c.stage(), Stage::IDLE);
    EXPECT_FALSE(c.pending().has_value());
    EXPECT_EQ(c.query(), 2500);
}

// The transaction state machine.

TEST(NPUWContinuation, PreflightAbortConsumesCommandAndKeepsCounters) {
    auto c = make_after_prefill(3000u, 20u);
    c.propose(3000);
    c.abort_preflight();
    EXPECT_EQ(c.stage(), Stage::IDLE);
    EXPECT_EQ(c.query(), 3020);
    EXPECT_FALSE(c.pending().has_value());
}

TEST(NPUWContinuation, PreflightAbortRequiresAPendingKeep) {
    // Only a pending keep may be aborted. A pending reset stays pending until the
    // full prompt arrives, and aborting a poisoned request must not clear the
    // poisoning that only reset() owns.
    auto idle = make_after_prefill(3000u);
    EXPECT_THROW(idle.abort_preflight(), ov::Exception);

    auto reset_pending = make_after_prefill(3000u);
    reset_pending.request_reset();
    EXPECT_THROW(reset_pending.abort_preflight(), ov::Exception);
    EXPECT_EQ(reset_pending.stage(), Stage::PENDING);
    EXPECT_EQ(reset_pending.pending(), 0u);

    auto poisoned = make_after_prefill(3000u);
    poisoned.propose(3000);
    poisoned.poison();
    EXPECT_THROW(poisoned.abort_preflight(), ov::Exception);
    EXPECT_EQ(poisoned.stage(), Stage::POISONED);
}

TEST(NPUWContinuation, FailureDuringInferencePoisonsTheRequest) {
    auto c = make_after_prefill(3000u);
    c.propose(3000);
    c.poison();
    EXPECT_EQ(c.stage(), Stage::POISONED);
    EXPECT_EQ(c.query(), 0);
    // Poisoned requests reject inference until reset() is called.
    EXPECT_THROW(c.pending(), ov::Exception);
}

TEST(NPUWContinuation, ResetRecoversAPoisonedRequest) {
    auto c = make_after_prefill(3000u);
    c.propose(3000);
    c.poison();
    c.request_reset();
    EXPECT_EQ(c.stage(), Stage::PENDING);
    EXPECT_EQ(c.pending(), 0u);
    // Idempotent while already pending.
    c.request_reset();
    EXPECT_EQ(c.stage(), Stage::PENDING);
    EXPECT_EQ(c.pending(), 0u);
}

TEST(NPUWContinuation, ResetDiscardsAPendingKeep) {
    auto c = make_after_prefill(3000u);
    c.propose(3000);
    c.request_reset();
    EXPECT_EQ(c.pending(), 0u);
    EXPECT_EQ(c.query(), 0);
}

TEST(NPUWContinuation, CommitAfterResetPrefillReturnsToIdle) {
    auto c = make_after_prefill(2048u);
    c.request_reset();
    c.commit_prefill(500u);
    EXPECT_EQ(c.stage(), Stage::IDLE);
    EXPECT_EQ(c.query(), 500);
    EXPECT_EQ(c.watermark(), 500u);
}

TEST(NPUWContinuation, CommitAndPublishRejectInvalidStates) {
    // A commit must never clear the poisoning that only reset() owns.
    auto poisoned = make_after_prefill(3000u);
    poisoned.propose(3000);
    poisoned.poison();
    EXPECT_THROW(poisoned.commit_prefill(3000u), ov::Exception);
    EXPECT_THROW(poisoned.publish_generate(3000u), ov::Exception);
    EXPECT_EQ(poisoned.stage(), Stage::POISONED);

    // Generate only runs when no command is pending, so a publish with an armed
    // command is a sequencing error. A commit from PENDING is the ordinary
    // transaction commit and stays legal.
    auto armed = make_after_prefill(3000u);
    armed.propose(3000);
    EXPECT_THROW(armed.publish_generate(3100u), ov::Exception);
    EXPECT_EQ(armed.stage(), Stage::PENDING);
    EXPECT_EQ(armed.pending(), 2048u);
}

// Watermark accounting.

TEST(NPUWContinuation, WatermarkFollowsPrefillOnly) {
    auto c = make_enabled();
    c.commit_prefill(2048u);
    EXPECT_EQ(c.watermark(), 2048u);
    c.publish_generate(2148u);
    EXPECT_EQ(c.watermark(), 2048u);
    EXPECT_EQ(c.live_tokens(), 2148u);
}

TEST(NPUWContinuation, SpeculativeTrimClampsWatermarkDown) {
    auto c = make_enabled();
    c.commit_prefill(2048u);
    // A trim below the watermark must lower it, or a later grant could point at
    // tokens that no longer exist.
    c.publish_generate(1500u);
    EXPECT_EQ(c.watermark(), 1500u);
    EXPECT_THROW(c.propose(2048), ov::Exception);
    EXPECT_EQ(c.propose(1500), 1024u);
}

TEST(NPUWContinuation, DisabledCoordinatorReportsNoCommand) {
    ContinuationCoordinator c;
    EXPECT_FALSE(c.enabled());
    EXPECT_FALSE(c.pending().has_value());
    EXPECT_THROW(c.propose(0), ov::Exception);
}

// The variable-state channel keeps its legacy contract without a coordinator.

TEST(NPUWStoredTokensState, LegacySetStateThrows) {
    auto state = std::make_shared<ov::npuw::StoredTokensState>();
    EXPECT_THROW(state->set_state(scalar_i64(1)), ov::Exception);
}

TEST(NPUWStoredTokensState, LegacyResetZeroes) {
    auto state = std::make_shared<ov::npuw::StoredTokensState>();
    state->reset();
    EXPECT_EQ(read_scalar(state->get_state()), 0);
}

}  // anonymous namespace
