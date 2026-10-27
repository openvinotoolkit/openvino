// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tp/collective_contract.hpp"

#include <gtest/gtest.h>

#include <thread>

#include "openvino/core/any.hpp"

using ov::tp::Blob;
using ov::tp::InProcessExchange;

namespace {
Blob blob_of(int rank, int round) {
    return Blob{static_cast<uint8_t>(rank), static_cast<uint8_t>(round)};
}
}  // namespace

TEST(TPCollectiveContractTest, EndpointsKnowTheirRank) {
    const auto group = InProcessExchange::create_group(3, std::chrono::milliseconds{1000});
    ASSERT_EQ(group.size(), 3u);
    for (int rank = 0; rank < 3; ++rank) {
        EXPECT_EQ(group[rank]->rank(), rank);
        EXPECT_EQ(group[rank]->world_size(), 3);
    }
}

// One thread publishes for every rank first and collects afterwards: the way the TP plugin creates contexts.
TEST(TPCollectiveContractTest, PublishAllThenCollectOnOneThread) {
    const auto group = InProcessExchange::create_group(4, std::chrono::milliseconds{1000});
    for (int rank = 0; rank < 4; ++rank) {
        group[rank]->publish(blob_of(rank, 0));
    }
    for (int rank = 0; rank < 4; ++rank) {
        const auto blobs = group[rank]->collect();
        ASSERT_EQ(blobs.size(), 4u);
        for (int peer = 0; peer < 4; ++peer) {
            EXPECT_EQ(blobs[peer], blob_of(peer, 0)) << "rank " << rank << " peer " << peer;
        }
    }
}

// A rank may run a round ahead of a peer that has not collected yet.
TEST(TPCollectiveContractTest, RoundsStaySeparateWhenRanksRunAhead) {
    const auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{1000});
    group[0]->publish(blob_of(0, 0));
    group[1]->publish(blob_of(1, 0));
    EXPECT_EQ(group[0]->collect(), (std::vector<Blob>{blob_of(0, 0), blob_of(1, 0)}));
    group[0]->publish(blob_of(0, 1));
    EXPECT_EQ(group[1]->collect(), (std::vector<Blob>{blob_of(0, 0), blob_of(1, 0)}));
    group[1]->publish(blob_of(1, 1));
    EXPECT_EQ(group[0]->collect(), (std::vector<Blob>{blob_of(0, 1), blob_of(1, 1)}));
    EXPECT_EQ(group[1]->collect(), (std::vector<Blob>{blob_of(0, 1), blob_of(1, 1)}));
}

TEST(TPCollectiveContractTest, ConcurrentRanksExchangeManyRounds) {
    constexpr int world_size = 4;
    constexpr int rounds = 50;
    const auto group = InProcessExchange::create_group(world_size, std::chrono::milliseconds{10000});
    std::vector<std::thread> threads;
    std::vector<int> mismatches(world_size, 0);
    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back([&, rank] {
            for (int round = 0; round < rounds; ++round) {
                group[rank]->publish(blob_of(rank, round));
                const auto blobs = group[rank]->collect();
                for (int peer = 0; peer < world_size; ++peer) {
                    mismatches[rank] += blobs[peer] != blob_of(peer, round) ? 1 : 0;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (int rank = 0; rank < world_size; ++rank) {
        EXPECT_EQ(mismatches[rank], 0) << "rank " << rank;
    }
}

TEST(TPCollectiveContractTest, CollectTimesOutWhenARankNeverPublishes) {
    const auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{50});
    group[0]->publish({});
    EXPECT_THROW(group[0]->collect(), ov::Exception);
}

TEST(TPCollectiveContractTest, RejectsCallsOutOfOrder) {
    const auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{50});
    EXPECT_THROW(group[0]->collect(), ov::Exception);
    group[0]->publish({});
    EXPECT_THROW(group[0]->publish({}), ov::Exception);
}

TEST(TPCollectiveContractTest, RejectsInvalidGroups) {
    EXPECT_THROW(InProcessExchange::create_group(0, std::chrono::milliseconds{50}), ov::Exception);
    EXPECT_THROW(InProcessExchange::create_group(2, std::chrono::milliseconds{-1}), ov::Exception);
}

TEST(TPCollectiveContractTest, PayloadHeaderLayout) {
    ov::tp::PayloadHeader header;
    EXPECT_EQ(header.magic, 0x5054564Fu);
    EXPECT_EQ(header.backend[0], '\0');
    EXPECT_EQ(header.version, 0u);
}

// The groups travel through an ov::AnyMap and may end up printed in a log.
TEST(TPCollectiveContractTest, GroupsSurviveAnAnyMap) {
    const auto endpoints = InProcessExchange::create_group(2, std::chrono::milliseconds{50});
    const std::map<int, ov::tp::CollectiveGroup> groups{{0, {endpoints[1], "L0_P2P"}}};
    const ov::AnyMap params{ov::tp::internal::groups(groups), ov::tp::internal::channels(1u)};

    const auto restored = params.at(ov::tp::internal::groups.name()).as<std::map<int, ov::tp::CollectiveGroup>>();
    ASSERT_EQ(restored.count(0), 1u);
    EXPECT_EQ(restored.at(0).exchange, endpoints[1]);
    EXPECT_EQ(restored.at(0).backend, "L0_P2P");
    EXPECT_EQ(ov::Any(params).as<std::string>().find("L0_P2P") != std::string::npos, true);
    EXPECT_EQ(ov::tp::CollectiveGroup{}.backend, ov::tp::auto_backend);
}
