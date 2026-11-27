// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/sparse_attn/paged_page_table_check.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "utils/plain_tensor.hpp"

namespace {

constexpr size_t kPageSize = 4;

// One sequence of 3 new tokens after 2 past ones: 5 tokens need 2 pages of 4 rows. `pages` are its block ids.
void check(std::vector<int32_t> pages, size_t num_pages, size_t page_size = kPageSize) {
    std::vector<int32_t> past = {2};
    std::vector<int32_t> subsequence_begins = {0, 3};
    std::vector<int32_t> block_indices_begins = {0, static_cast<int32_t>(pages.size())};
    ov::intel_cpu::PlainTensor past_lens, subsequence, block_indices, block_begins;
    past_lens.resize<int32_t>({1}, past.data());
    subsequence.resize<int32_t>({2}, subsequence_begins.data());
    block_indices.resize<int32_t>({pages.size()}, pages.data());
    block_begins.resize<int32_t>({2}, block_indices_begins.data());
    ov::intel_cpu::check_paged_page_table(past_lens, subsequence, block_indices, block_begins, 3, page_size, num_pages);
}

// The flattened slots (num_pages * page_size of them) must fit int32: with 4-row pages, 2^29 pages is the most.
constexpr size_t kMaxPagesOfInt32Slots = (size_t{1} << 31) / kPageSize;

TEST(PagedPageTableCheckTest, AcceptsBlockIdsInRange) {
    EXPECT_NO_THROW(check({0, 1}, 2));
    EXPECT_NO_THROW(check({5, 0}, 6));
    EXPECT_NO_THROW(check({7, 3}, kMaxPagesOfInt32Slots));
}

TEST(PagedPageTableCheckTest, RejectsBlockIdEqualToPageCount) {
    EXPECT_THROW(check({0, 2}, 2), ov::Exception);
    EXPECT_THROW(check({6, 0}, 6), ov::Exception);
}

TEST(PagedPageTableCheckTest, RejectsNegativeBlockIds) {
    EXPECT_THROW(check({0, -1}, 2), ov::Exception);
    EXPECT_THROW(check({std::numeric_limits<int32_t>::min(), 0}, 2), ov::Exception);
    EXPECT_THROW(check({0, -1}, kMaxPagesOfInt32Slots), ov::Exception);
}

TEST(PagedPageTableCheckTest, AcceptsTheLargestBlockIdsOfTheSlotRange) {
    // 1-row pages: 2^31 slots, so block ids up to INT32_MAX are in range
    EXPECT_NO_THROW(check({std::numeric_limits<int32_t>::max(), 0, 1, 2, 3}, size_t{1} << 31, 1));
    EXPECT_NO_THROW(check({static_cast<int32_t>(kMaxPagesOfInt32Slots - 1), 0}, kMaxPagesOfInt32Slots));
}

// Regression: SparsePA stores a flattened slot (block * page_size + offset) in an int32. With 32-row pages the block
// 2^26 gives the slot 2^31, which wraps to INT32_MIN and the K/V write is skipped.
TEST(PagedPageTableCheckTest, RejectsFlattenedSlotsBeyondInt32) {
    constexpr size_t kTokensPerPage = 32;
    constexpr size_t kLimit = (size_t{1} << 31) / kTokensPerPage;  // 2^26 pages: the largest slot is 2^31 - 1
    EXPECT_NO_THROW(check({static_cast<int32_t>(kLimit - 1), 0}, kLimit, kTokensPerPage));
    EXPECT_THROW(check({static_cast<int32_t>(kLimit), 0}, kLimit + 1, kTokensPerPage), ov::Exception);
    EXPECT_THROW(check({0, 1}, kLimit + 1, kTokensPerPage), ov::Exception);
    // 1-row pages: the 5 tokens need 5 pages
    EXPECT_NO_THROW(check({0, 1, 2, 3, 4}, size_t{1} << 31, 1));
    EXPECT_THROW(check({0, 1, 2, 3, 4}, (size_t{1} << 31) + 1, 1), ov::Exception);
}

TEST(PagedPageTableCheckTest, ReportsTheBlockId) {
    try {
        check({0, 9}, 2);
        FAIL() << "expected an exception";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("block id 9"), std::string::npos) << ex.what();
    }
}

}  // namespace
