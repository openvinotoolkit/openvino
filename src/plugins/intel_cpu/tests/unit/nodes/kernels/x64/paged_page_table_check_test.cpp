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
void check(std::vector<int32_t> pages, size_t num_pages) {
    std::vector<int32_t> past = {2};
    std::vector<int32_t> subsequence_begins = {0, 3};
    std::vector<int32_t> block_indices_begins = {0, static_cast<int32_t>(pages.size())};
    ov::intel_cpu::PlainTensor past_lens, subsequence, block_indices, block_begins;
    past_lens.resize<int32_t>({1}, past.data());
    subsequence.resize<int32_t>({2}, subsequence_begins.data());
    block_indices.resize<int32_t>({pages.size()}, pages.data());
    block_begins.resize<int32_t>({2}, block_indices_begins.data());
    ov::intel_cpu::check_paged_page_table(past_lens, subsequence, block_indices, block_begins, 3, kPageSize, num_pages);
}

TEST(PagedPageTableCheckTest, AcceptsBlockIdsInRange) {
    EXPECT_NO_THROW(check({0, 1}, 2));
    EXPECT_NO_THROW(check({5, 0}, 6));
    EXPECT_NO_THROW(check({7, 3}, std::numeric_limits<size_t>::max()));
}

TEST(PagedPageTableCheckTest, RejectsBlockIdEqualToPageCount) {
    EXPECT_THROW(check({0, 2}, 2), ov::Exception);
    EXPECT_THROW(check({6, 0}, 6), ov::Exception);
}

TEST(PagedPageTableCheckTest, RejectsNegativeBlockIds) {
    EXPECT_THROW(check({0, -1}, 2), ov::Exception);
    EXPECT_THROW(check({std::numeric_limits<int32_t>::min(), 0}, 2), ov::Exception);
    // also with a page count above 2^31, where a wrapped negative id must still be out of range
    EXPECT_THROW(check({0, -1}, size_t{1} << 40), ov::Exception);
}

TEST(PagedPageTableCheckTest, AcceptsLargeNonNegativeBlockIdsOfHugeCaches) {
    EXPECT_NO_THROW(check({std::numeric_limits<int32_t>::max(), 0}, size_t{1} << 40));
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
