// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/core/except.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu {

/// @brief Validates the PagedAttention-style page table of a call once, in O(sequences + page table entries), so that
/// the kernels can index the caches without bounds checks.
///
/// Checked: past_lens >= 0; subsequence_begins starts at 0, never decreases and ends at `num_tokens` (every token
/// belongs to a sequence); block_indices_begins starts at >= 0, never decreases and stays within the block_indices
/// tensor; every sequence fits the pages it owns (past + q <= pages * page_size); every used block id is below
/// `num_pages`; the flattened slots (block * page_size + offset, below num_pages * page_size) fit int32, the type of
/// the slot mappings of the kernels.
///
/// @param num_tokens  tokens of the call (the rows of the per-token inputs)
/// @param page_size   tokens per page of the caches
/// @param num_pages   pages of the caches (the smallest outer dim of the caches addressed through the table)
inline void check_paged_page_table(const PlainTensor& past_lens,
                                   const PlainTensor& subsequence_begins,
                                   const PlainTensor& block_indices,
                                   const PlainTensor& block_indices_begins,
                                   size_t num_tokens,
                                   size_t page_size,
                                   size_t num_pages) {
    OPENVINO_ASSERT(subsequence_begins.size(0) >= 1 && block_indices_begins.size(0) == subsequence_begins.size(0),
                    "[CPU] paged attention: subsequence_begins and block_indices_begins must have one entry per "
                    "sequence plus one");
    const size_t num_seq = subsequence_begins.size(0) - 1;
    OPENVINO_ASSERT(past_lens.size(0) >= num_seq, "[CPU] paged attention: past_lens has fewer entries than sequences");
    OPENVINO_ASSERT(page_size > 0, "[CPU] paged attention: the page size must be positive");
    OPENVINO_ASSERT(num_pages <= (uint64_t{1} << 31) / page_size,
                    "[CPU] paged attention: ",
                    num_pages,
                    " pages of ",
                    page_size,
                    " tokens have slots beyond the int32 range");
    const auto* past = past_lens.ptr<int32_t>();
    const auto* token_begins = subsequence_begins.ptr<int32_t>();
    const auto* page_begins = block_indices_begins.ptr<int32_t>();
    const auto* pages = block_indices.ptr<int32_t>();
    OPENVINO_ASSERT(page_begins[0] >= 0, "[CPU] paged attention: block_indices_begins must start at >= 0");
    for (size_t seq = 0; seq < num_seq; seq++) {
        const int64_t q_len = static_cast<int64_t>(token_begins[seq + 1]) - token_begins[seq];
        const int64_t seq_pages = static_cast<int64_t>(page_begins[seq + 1]) - page_begins[seq];
        OPENVINO_ASSERT(q_len >= 0, "[CPU] paged attention: subsequence_begins must not decrease at sequence ", seq);
        OPENVINO_ASSERT(seq_pages >= 0,
                        "[CPU] paged attention: block_indices_begins must not decrease at sequence ",
                        seq);
        OPENVINO_ASSERT(past[seq] >= 0, "[CPU] paged attention: past_lens[", seq, "] is negative");
        OPENVINO_ASSERT(static_cast<int64_t>(past[seq]) + q_len <= seq_pages * static_cast<int64_t>(page_size),
                        "[CPU] paged attention: sequence ",
                        seq,
                        " does not fit its ",
                        seq_pages,
                        " cache pages");
    }
    OPENVINO_ASSERT(token_begins[0] == 0 && static_cast<size_t>(token_begins[num_seq]) == num_tokens,
                    "[CPU] paged attention: subsequence_begins must cover the tokens of the call");
    OPENVINO_ASSERT(static_cast<size_t>(page_begins[num_seq]) <= block_indices.size(0),
                    "[CPU] paged attention: block_indices_begins exceeds the size of block_indices");
    const auto page_limit = static_cast<uint32_t>(num_pages);  // at most 2^31, see the slot check above
    const int32_t first_page = page_begins[0];
    const int32_t last_page = page_begins[num_seq];
    uint32_t out_of_range = 0;
    for (int32_t i = first_page; i < last_page; i++) {
        out_of_range |= static_cast<uint32_t>(static_cast<uint32_t>(pages[i]) >= page_limit);
    }
    if (out_of_range != 0) {
        for (int32_t i = first_page; i < last_page; i++) {
            OPENVINO_ASSERT(static_cast<uint32_t>(pages[i]) < page_limit,
                            "[CPU] paged attention: block id ",
                            pages[i],
                            " is out of the cache's ",
                            num_pages,
                            " pages");
        }
    }
}

}  // namespace ov::intel_cpu
