// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_dependency_set.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using cldnn::memory_dependency_set;

namespace {

void check_insertions(const std::vector<uint32_t>& ids, bool read_during_construction = false) {
    memory_dependency_set actual;
    std::set<uint32_t> expected;
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto id = ids[i];
        ASSERT_EQ(actual.contains(id), expected.count(id) != 0);
        actual.insert(id);
        expected.insert(id);
        ASSERT_TRUE(actual.contains(id));
        ASSERT_EQ(actual.contains(id ^ 1), expected.count(id ^ 1) != 0);
        if (read_during_construction && i % 127 == 0) {
            ASSERT_EQ(actual.values(), std::vector<uint32_t>(expected.begin(), expected.end()));
        }
    }
    const std::vector<uint32_t> result(expected.begin(), expected.end());
    EXPECT_EQ(actual.values(), result);
    EXPECT_EQ(actual.values(), result);
    for (const auto id : ids)
        EXPECT_TRUE(actual.contains(id));
}

TEST(memory_dependency_set, empty) {
    memory_dependency_set deps;
    EXPECT_FALSE(deps.contains(0));
    EXPECT_FALSE(deps.contains(std::numeric_limits<uint32_t>::max()));
    EXPECT_TRUE(deps.values().empty());
}

TEST(memory_dependency_set, dense_insertion_orders_and_duplicates) {
    std::vector<uint32_t> ids(8192);
    std::iota(ids.begin(), ids.end(), 100000);
    check_insertions(ids);
    std::reverse(ids.begin(), ids.end());
    check_insertions(ids);
    std::mt19937 rng(42);
    std::shuffle(ids.begin(), ids.end(), rng);
    auto repeated = ids;
    repeated.insert(repeated.end(), ids.begin(), ids.end());
    check_insertions(repeated);
}

TEST(memory_dependency_set, sparse_full_uint32_range) {
    std::vector<uint32_t> ids{0, 63, 64, 65, std::numeric_limits<uint32_t>::max()};
    std::mt19937 rng(51);
    for (size_t i = 0; i < 4096; ++i)
        ids.push_back(rng());
    check_insertions(ids);
}

TEST(memory_dependency_set, dense_near_uint32_limits) {
    std::vector<uint32_t> ids(4096);
    std::iota(ids.begin(), ids.end(), std::numeric_limits<uint32_t>::max() - 4095);
    check_insertions(ids);
    std::reverse(ids.begin(), ids.end());
    check_insertions(ids);
    std::iota(ids.begin(), ids.end(), 0);
    check_insertions(ids);
}

TEST(memory_dependency_set, sparse_outliers_after_dense_insertion) {
    std::vector<uint32_t> ids(8192);
    std::iota(ids.begin(), ids.end(), 10000);
    ids.push_back(0);
    ids.push_back(std::numeric_limits<uint32_t>::max());
    ids.push_back(10001);
    ids.push_back(20000);
    check_insertions(ids);
}

TEST(memory_dependency_set, materialize_then_continue_building) {
    std::vector<uint32_t> ids(4096);
    std::iota(ids.begin(), ids.end(), 10000);
    std::mt19937 rng(123);
    std::shuffle(ids.begin(), ids.end(), rng);
    check_insertions(ids, true);
}

TEST(memory_dependency_set, load_existing_sorted_vector_and_continue) {
    memory_dependency_set deps(std::vector<uint32_t>{0, 64, 128, 1024});
    for (uint32_t i = 0; i < 2048; ++i)
        deps.insert(i);
    std::vector<uint32_t> expected(2048);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(deps.values(), expected);
}

TEST(memory_dependency_set, copy_during_construction_is_independent) {
    memory_dependency_set original;
    for (uint32_t i = 100; i < 4196; ++i)
        original.insert(i);
    auto copy = original;
    copy.insert(99);
    original.insert(4196);
    EXPECT_TRUE(copy.contains(99));
    EXPECT_FALSE(original.contains(99));
    EXPECT_FALSE(copy.contains(4196));
    EXPECT_TRUE(original.contains(4196));
    EXPECT_EQ(copy.values().size(), 4097);
    EXPECT_EQ(original.values().size(), 4097);
}

TEST(memory_dependency_set, alternating_dense_clusters_and_outliers) {
    std::vector<uint32_t> ids;
    for (uint32_t cluster = 0; cluster < 4; ++cluster) {
        for (uint32_t i = 0; i < 2048; ++i)
            ids.push_back(cluster * 1000000000u + i);
    }
    check_insertions(ids);
}

TEST(memory_dependency_set, randomized_membership_after_every_addition) {
    memory_dependency_set actual;
    std::set<uint32_t> expected;
    std::mt19937 rng(2026);
    for (size_t i = 0; i < 20000; ++i) {
        const uint32_t id = rng() % 4096;
        actual.insert(id);
        expected.insert(id);
        for (size_t j = 0; j < 4; ++j) {
            const uint32_t query = rng() % 8192;
            ASSERT_EQ(actual.contains(query), expected.count(query) != 0);
        }
    }
    EXPECT_EQ(actual.values(), std::vector<uint32_t>(expected.begin(), expected.end()));
}

}  // namespace
