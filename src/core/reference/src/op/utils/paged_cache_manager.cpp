// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/paged_cache_manager.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "openvino/core/except.hpp"

namespace ov {
namespace reference {
namespace paged_attention_cache {

namespace {
inline std::size_t safe_mul(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > (std::numeric_limits<std::size_t>::max() / b)) {
        OPENVINO_THROW("PagedCacheManager: tensor size overflow");
    }
    return a * b;
}
}  // namespace

PagedCacheManager::PagedCacheManager(ov::element::Type elem_type,
                                     EvictionPolicy policy,
                                     std::size_t max_cache_bytes,
                                     float attention_mass_p)
    : m_elem_type(elem_type),
      m_policy(policy),
      m_max_cache_bytes(max_cache_bytes),
      m_attention_mass_p(attention_mass_p) {}

std::size_t PagedCacheManager::elem_bytes_or_throw(ov::element::Type et) {
    const auto sz = et.size();
    if (sz == 0) {
        OPENVINO_THROW("PagedCacheManager: unsupported element type for cache");
    }
    return sz;
}

void PagedCacheManager::validate_cache_rank4_or_throw(const ov::Shape& key_cache_shape,
                                                      const ov::Shape& value_cache_shape) {
    if (key_cache_shape.size() != 4 || value_cache_shape.size() != 4) {
        OPENVINO_THROW("PagedCacheManager: reference cache manager expects key_cache/value_cache rank 4 ",
                       "[num_blocks, num_kv_heads, block_size, head_size]. Got ranks ",
                       key_cache_shape.size(),
                       " and ",
                       value_cache_shape.size());
    }
}

std::size_t PagedCacheManager::tensor_byte_size(const ov::Shape& shape, std::size_t elem_bytes) {
    std::size_t prod = 1;
    for (const auto d : shape) {
        prod = safe_mul(prod, static_cast<std::size_t>(d));
    }
    return safe_mul(prod, elem_bytes);
}

void PagedCacheManager::parse_cache_layout_or_throw(OperatorState& st,
                                                    const ov::Shape& key_cache_shape,
                                                    const ov::Shape& value_cache_shape,
                                                    std::size_t elem_bytes) {
    validate_cache_rank4_or_throw(key_cache_shape, value_cache_shape);

    st.num_blocks = static_cast<std::size_t>(key_cache_shape[0]);
    st.num_kv_heads = static_cast<std::size_t>(key_cache_shape[1]);
    st.block_size = static_cast<std::size_t>(key_cache_shape[2]);
    st.key_head_size = static_cast<std::size_t>(key_cache_shape[3]);

    if (value_cache_shape[0] != key_cache_shape[0] || value_cache_shape[1] != key_cache_shape[1] ||
        value_cache_shape[2] != key_cache_shape[2]) {
        OPENVINO_THROW("PagedCacheManager: key_cache and value_cache layout mismatch");
    }
    st.value_head_size = static_cast<std::size_t>(value_cache_shape[3]);

    if (st.num_blocks == 0 || st.num_kv_heads == 0 || st.block_size == 0) {
        OPENVINO_THROW("PagedCacheManager: cache shape has zero dimension");
    }

    st.key_block_bytes = safe_mul(safe_mul(st.num_kv_heads, st.block_size), safe_mul(st.key_head_size, elem_bytes));
    st.value_block_bytes = safe_mul(safe_mul(st.num_kv_heads, st.block_size), safe_mul(st.value_head_size, elem_bytes));
}

PagedCacheManager::OperatorState& PagedCacheManager::get_state(std::uintptr_t node_key) {
    auto it = m_ops.find(node_key);
    if (it == m_ops.end()) {
        OPENVINO_THROW("PagedCacheManager: operator state not found for node_key");
    }
    return it->second;
}

const PagedCacheManager::OperatorState& PagedCacheManager::get_state(std::uintptr_t node_key) const {
    auto it = m_ops.find(node_key);
    if (it == m_ops.end()) {
        OPENVINO_THROW("PagedCacheManager: operator state not found for node_key");
    }
    return it->second;
}

void PagedCacheManager::init_sequences_from_block_tables(OperatorState& st,
                                                         const std::int32_t* block_indices,
                                                         std::size_t block_indices_count,
                                                         const std::int32_t* block_indices_begins,
                                                         std::size_t begins_count,
                                                         const std::int32_t* past_lens,
                                                         std::size_t past_lens_count) {
    const std::size_t seq_count = past_lens_count;
    st.sequences.assign(seq_count, SequenceState{});

    // If no block tables were provided, initialize empty sequences
    if (block_indices == nullptr || block_indices_begins == nullptr || begins_count != (seq_count + 1)) {
        for (std::size_t s = 0; s < seq_count; ++s) {
            st.sequences[s].logical_length = past_lens ? past_lens[s] : 0;
        }
        return;
    }

    // Fill per-sequence block lists based on provided tables
    for (std::size_t s = 0; s < seq_count; ++s) {
        const std::int32_t begin = block_indices_begins[s];
        const std::int32_t end = block_indices_begins[s + 1];
        if (begin < 0 || end < begin) {
            continue;
        }
        const std::size_t ubegin = static_cast<std::size_t>(begin);
        const std::size_t uend = static_cast<std::size_t>(end);
        if (ubegin > block_indices_count || uend > block_indices_count) {
            continue;
        }

        auto& seq = st.sequences[s];
        seq.logical_length = past_lens ? past_lens[s] : 0;

        for (std::size_t i = ubegin; i < uend; ++i) {
            const std::int32_t bid = block_indices[i];
            if (bid >= 0) {
                seq.blocks.push_back(bid);
                if (static_cast<std::size_t>(bid) < st.block_used.size()) {
                    st.block_used[static_cast<std::size_t>(bid)] = 1;
                }
            }
        }

        // Ensure trim_front is block-aligned and within logical length
        seq.trim_front = 0;
    }
}

std::int32_t PagedCacheManager::steal_block_fifo(OperatorState& st, std::size_t requester_seq) {
    // Take the oldest block from the sequence with the most blocks
    std::size_t best_seq = std::numeric_limits<std::size_t>::max();
    std::size_t best_blocks = 0;

    for (std::size_t s = 0; s < st.sequences.size(); ++s) {
        if (s == requester_seq) {
            // prefer not to steal from the requester if possible
            continue;
        }
        const auto& seq = st.sequences[s];
        if (seq.blocks.empty()) {
            continue;
        }
        const std::size_t blocks = seq.blocks.size();
        if (blocks > best_blocks) {
            best_blocks = blocks;
            best_seq = s;
        }
    }

    if (best_seq == std::numeric_limits<std::size_t>::max()) {
        // as a last resort, steal from the requester itself
        best_seq = requester_seq;
    }

    auto& victim = st.sequences[best_seq];
    if (victim.blocks.empty()) {
        OPENVINO_THROW("PagedCacheManager: cannot steal a block (no victim blocks)");
    }

    const std::int32_t bid = victim.blocks.front();
    victim.blocks.pop_front();
    if (!victim.block_scores.empty()) {
        victim.block_scores.pop_front();
    }
    victim.diversity_matrix.clear();
    victim.diversity_n_blocks = 0;
    victim.trim_front += static_cast<std::int32_t>(st.block_size);
    return bid;
}

std::int32_t PagedCacheManager::steal_block_by_score(OperatorState& st, std::size_t requester_seq) {
    // Pick the sequence whose front block has the lowest accumulated attention score
    // We only evict front blocks so that the deque-index-to-token-position mapping stays valid
    // Falls back to FIFO if no scores have been recorded yet
    std::size_t best_seq = std::numeric_limits<std::size_t>::max();
    float best_score = std::numeric_limits<float>::max();
    bool found_any = false;

    for (std::size_t s = 0; s < st.sequences.size(); ++s) {
        auto& seq = st.sequences[s];
        if (seq.blocks.empty() || seq.block_scores.empty()) {
            continue;
        }
        float penalty = (s == requester_seq) ? 1e12f : 0.f;
        float score = seq.block_scores.front() + penalty;
        if (score < best_score) {
            best_score = score;
            best_seq = s;
            found_any = true;
        }
    }

    if (!found_any) {
        // no scores recorded yet, fall back to FIFO
        return steal_block_fifo(st, requester_seq);
    }

    auto& victim = st.sequences[best_seq];
    const std::int32_t bid = victim.blocks.front();
    victim.blocks.pop_front();
    if (!victim.block_scores.empty()) {
        victim.block_scores.pop_front();
    }
    victim.diversity_matrix.clear();
    victim.diversity_n_blocks = 0;
    victim.trim_front += static_cast<std::int32_t>(st.block_size);
    return bid;
}

std::int32_t PagedCacheManager::steal_block_by_diversity(OperatorState& st, std::size_t requester_seq) {
    // Full Adaptive R-KV eviction:
    // 1. Identify the "retained set" of blocks per sequence via attention-mass gating.
    // 2. For each non-retained front block, compute diversity as mean over columns
    //    corresponding to retained blocks' tokens only.
    // 3. Evict the non-retained front block with the lowest filtered diversity.
    //
    // Falls back to score eviction if no diversity data was fed

    struct Candidate {
        std::size_t seq;
        float filtered_div;
    };
    std::vector<Candidate> candidates;

    for (std::size_t s = 0; s < st.sequences.size(); ++s) {
        auto& seq = st.sequences[s];
        if (seq.blocks.empty()) {
            continue;
        }

        // Need diversity matrix to participate
        if (seq.diversity_matrix.empty() || seq.diversity_n_blocks == 0 || seq.diversity_evict_size == 0) {
            continue;
        }

        // Step 1: Attention-mass gating within this sequence's eviction zone.
        // Determine which blocks in the eviction zone are in the "retained set"
        const std::size_t n_blks = seq.diversity_n_blocks;
        const std::size_t evict_sz = seq.diversity_evict_size;
        const std::size_t start_blk = seq.diversity_start_block;

        // Gather per-block attention scores for the eviction zone
        std::vector<std::pair<float, std::size_t>> scored_blocks;  // (score, zone_index)
        scored_blocks.reserve(n_blks);
        for (std::size_t i = 0; i < n_blks; ++i) {
            float sc = 0.f;
            const std::size_t deque_idx = start_blk + i;
            if (deque_idx < seq.block_scores.size()) {
                sc = seq.block_scores[deque_idx];
            }
            scored_blocks.emplace_back(sc, i);
        }

        // Sort descending by score
        std::sort(scored_blocks.begin(), scored_blocks.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Greedily select blocks covering >= p fraction of total attention mass
        float total_score = 0.f;
        for (auto& sb : scored_blocks)
            total_score += sb.first;
        const float target = total_score * m_attention_mass_p;

        std::vector<bool> retained(n_blks, false);
        float cumsum = 0.f;
        for (auto& sb : scored_blocks) {
            if (cumsum >= target && target > 0.f)
                break;
            retained[sb.second] = true;
            cumsum += sb.first;
        }

        // The front block of the deque corresponds to zone index (0 - start_blk) if start_blk==0,
        // or may be outside the eviction zone entirely.
        // Find which zone index the deque front block maps to:
        // deque index 0 -> zone index = 0 - start_blk (only valid if start_blk == 0)
        // If the front block is in the start area (before the eviction zone), it is
        // not in the retained set and is eligible for eviction
        bool front_is_retained = false;
        if (start_blk == 0 && n_blks > 0) {
            front_is_retained = retained[0];
        }

        // If the front block is retained (attention-important), skip this sequence
        if (front_is_retained) {
            continue;
        }

        // Step 2: Compute filtered diversity for the front block.
        // diversity_matrix row 0 corresponds to the first eviction zone block.
        // If start_blk > 0, the front block of the deque is actually in the "start area"
        // and has no diversity row - use 0 diversity (always evictable)
        float div_score = 0.f;
        if (start_blk == 0 && n_blks > 0) {
            // Front block is zone block 0; its row is diversity_matrix[0, :].
            // Mean over columns corresponding to retained blocks' tokens only
            float sum = 0.f;
            std::size_t count = 0;
            for (std::size_t bi = 0; bi < n_blks; ++bi) {
                if (!retained[bi])
                    continue;
                // Columns for block bi: [bi * block_size, (bi+1) * block_size)
                const std::size_t col_start = bi * st.block_size;
                const std::size_t col_end = std::min(col_start + st.block_size, evict_sz);
                for (std::size_t c = col_start; c < col_end; ++c) {
                    if (c < evict_sz) {
                        sum += seq.diversity_matrix[0 * evict_sz + c];
                        count++;
                    }
                }
            }
            div_score = (count > 0) ? (sum / static_cast<float>(count)) : 0.f;
        }

        float penalty = (s == requester_seq) ? 1e12f : 0.f;
        candidates.push_back({s, div_score + penalty});
    }

    if (candidates.empty()) {
        // no diversity data available, fall back to score eviction
        return steal_block_by_score(st, requester_seq);
    }

    // Pick candidate with lowest filtered diversity
    std::size_t best_seq = candidates[0].seq;
    float best_div = candidates[0].filtered_div;
    for (std::size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].filtered_div < best_div) {
            best_div = candidates[i].filtered_div;
            best_seq = candidates[i].seq;
        }
    }

    auto& victim = st.sequences[best_seq];
    const std::int32_t bid = victim.blocks.front();
    victim.blocks.pop_front();
    if (!victim.block_scores.empty()) {
        victim.block_scores.pop_front();
    }
    victim.diversity_matrix.clear();
    victim.diversity_n_blocks = 0;
    victim.trim_front += static_cast<std::int32_t>(st.block_size);
    return bid;
}

std::int32_t PagedCacheManager::steal_block(OperatorState& st, std::size_t requester_seq) {
    switch (m_policy) {
    case EvictionPolicy::SCORE:
        return steal_block_by_score(st, requester_seq);
    case EvictionPolicy::ADAPTIVE_RKV:
        return steal_block_by_diversity(st, requester_seq);
    case EvictionPolicy::FIFO:
    default:
        return steal_block_fifo(st, requester_seq);
    }
}

std::int32_t PagedCacheManager::allocate_block(OperatorState& st, std::size_t requester_seq) {
    if (m_max_cache_bytes > 0) {
        const std::size_t bytes_per_block = st.key_block_bytes + st.value_block_bytes;
        const std::size_t max_blocks =
            (bytes_per_block > 0) ? std::max<std::size_t>(1, m_max_cache_bytes / bytes_per_block) : st.num_blocks;
        const std::size_t active = st.num_blocks - st.free_blocks.size();
        if (active >= max_blocks) {
            return steal_block(st, requester_seq);
        }
    }

    if (!st.free_blocks.empty()) {
        const std::int32_t bid = st.free_blocks.back();
        st.free_blocks.pop_back();
        if (bid >= 0 && static_cast<std::size_t>(bid) < st.block_used.size()) {
            st.block_used[static_cast<std::size_t>(bid)] = 1;
        }
        return bid;
    }
    // No free blocks left: evict one block from a victim sequence
    return steal_block(st, requester_seq);
}

bool PagedCacheManager::ensure_operator(std::uintptr_t node_key,
                                        const void* key_cache_init,
                                        const void* value_cache_init,
                                        const ov::Shape& key_cache_shape,
                                        const ov::Shape& value_cache_shape,
                                        const std::int32_t* block_indices_init,
                                        std::size_t block_indices_count,
                                        const std::int32_t* block_indices_begins_init,
                                        std::size_t block_indices_begins_count,
                                        const std::int32_t* past_lens_init,
                                        std::size_t past_lens_count) {
    if (m_ops.find(node_key) != m_ops.end()) {
        return false;
    }

    OperatorState st;
    const std::size_t elem_bytes = elem_bytes_or_throw(m_elem_type);
    parse_cache_layout_or_throw(st, key_cache_shape, value_cache_shape, elem_bytes);

    // Allocate and copy initial cache tensors (once).
    const std::size_t key_bytes = tensor_byte_size(key_cache_shape, elem_bytes);
    const std::size_t value_bytes = tensor_byte_size(value_cache_shape, elem_bytes);
    st.key_cache = ov::AlignedBuffer{key_bytes};
    st.value_cache = ov::AlignedBuffer{value_bytes};

    if (key_cache_init) {
        std::memcpy(st.key_cache.get_ptr(), key_cache_init, key_bytes);
    } else {
        std::memset(st.key_cache.get_ptr(), 0, key_bytes);
    }

    if (value_cache_init) {
        std::memcpy(st.value_cache.get_ptr(), value_cache_init, value_bytes);
    } else {
        std::memset(st.value_cache.get_ptr(), 0, value_bytes);
    }

    // Initialize free list and per-seq state
    st.block_used.assign(st.num_blocks, 0);

    init_sequences_from_block_tables(st,
                                     block_indices_init,
                                     block_indices_count,
                                     block_indices_begins_init,
                                     block_indices_begins_count,
                                     past_lens_init,
                                     past_lens_count);

    // Build free block stack
    st.free_blocks.reserve(st.num_blocks);
    for (std::size_t b = 0; b < st.num_blocks; ++b) {
        if (!st.block_used[b]) {
            st.free_blocks.push_back(static_cast<std::int32_t>(b));
        }
    }

    m_ops.emplace(node_key, std::move(st));
    return true;
}

void PagedCacheManager::begin_step(std::uintptr_t node_key, const std::int32_t* past_lens, std::size_t seq_count) {
    auto& st = get_state(node_key);
    if (seq_count == 0) {
        OPENVINO_THROW("PagedCacheManager::begin_step: seq_count is 0");
    }
    if (st.sequences.size() != seq_count) {
        // For reference: allow resizing on first call (or if graph shape changes)
        st.sequences.assign(seq_count, SequenceState{});
    }

    for (std::size_t s = 0; s < seq_count; ++s) {
        const std::int32_t new_len = past_lens ? past_lens[s] : 0;
        auto& seq = st.sequences[s];

        // If the external timeline was reset/truncated, reset this sequence state
        if (new_len < seq.logical_length) {
            for (const std::int32_t bid : seq.blocks) {
                if (bid >= 0 && static_cast<std::size_t>(bid) < st.num_blocks && st.block_used[bid]) {
                    st.block_used[bid] = 0;
                    st.free_blocks.push_back(bid);
                }
            }
            seq.blocks.clear();
            seq.block_scores.clear();
            seq.diversity_matrix.clear();
            seq.diversity_n_blocks = 0;
            seq.diversity_evict_size = 0;
            seq.diversity_start_block = 0;
            seq.trim_front = 0;
            seq.logical_length = new_len;
            continue;
        }

        seq.logical_length = new_len;

        // If past_lens exceeds what the current retained blocks can represent,
        // do nothing here: the caller will append new tokens and ensure_token will allocate as needed
        // Also, if the sequence had previously been trimmed, keep trim_front
        if (seq.trim_front > seq.logical_length) {
            seq.trim_front = std::max<std::int32_t>(0, seq.logical_length - (std::int32_t)(st.block_size));
            seq.trim_front = (seq.trim_front / (std::int32_t)st.block_size) * (std::int32_t)st.block_size;
        }
    }
}

PagedCacheManager::TokenAddress PagedCacheManager::ensure_token(std::uintptr_t node_key,
                                                                std::size_t seq_idx,
                                                                std::int32_t token_pos) {
    auto& st = get_state(node_key);
    if (seq_idx >= st.sequences.size()) {
        OPENVINO_THROW("PagedCacheManager::ensure_token: seq_idx out of range");
    }

    auto& seq = st.sequences[seq_idx];

    if (token_pos < 0) {
        OPENVINO_THROW("PagedCacheManager::ensure_token: token_pos is negative");
    }

    // If token_pos is behind the trimmed window, it cannot be represented
    if (token_pos < seq.trim_front) {
        return TokenAddress{-1, 0};
    }

    const std::int32_t bs = static_cast<std::int32_t>(st.block_size);
    // Recompute relative position after any evictions that may have advanced trim_front
    // This is needed because stealing from the requester's own front block shifts trim_front
    auto recompute = [&]() -> std::pair<std::int32_t, std::int32_t> {
        const std::int32_t r = token_pos - seq.trim_front;
        const std::int32_t bi = (bs > 0) ? (r / bs) : 0;
        return {bi, (bs > 0) ? (r % bs) : 0};
    };

    auto [block_index, off] = recompute();

    // guard against infinite loops if the pool is too small for the requested position
    std::size_t max_iters = st.num_blocks + 1;
    while (seq.blocks.size() <= static_cast<std::size_t>(block_index) && max_iters-- > 0) {
        const std::int32_t bid = allocate_block(st, seq_idx);
        seq.blocks.push_back(bid);
        // trim_front may have changed if we stole from ourselves
        std::tie(block_index, off) = recompute();
    }

    if (seq.blocks.size() <= static_cast<std::size_t>(block_index)) {
        return TokenAddress{-1, 0};
    }

    return TokenAddress{seq.blocks[static_cast<std::size_t>(block_index)], off};
}

bool PagedCacheManager::resolve_token(std::uintptr_t node_key,
                                      std::size_t seq_idx,
                                      std::int32_t token_pos,
                                      TokenAddress& out_addr) const {
    const auto& st = get_state(node_key);
    if (seq_idx >= st.sequences.size()) {
        return false;
    }
    const auto& seq = st.sequences[seq_idx];

    if (token_pos < seq.trim_front || token_pos < 0 || token_pos >= seq.logical_length) {
        return false;
    }

    const std::int32_t rel = token_pos - seq.trim_front;
    const std::int32_t bs = static_cast<std::int32_t>(st.block_size);
    if (bs <= 0) {
        return false;
    }

    const std::int32_t block_index = rel / bs;
    const std::int32_t off = rel % bs;

    if (block_index < 0 || static_cast<std::size_t>(block_index) >= seq.blocks.size()) {
        return false;
    }

    out_addr.block = seq.blocks[static_cast<std::size_t>(block_index)];
    out_addr.offset = off;
    return out_addr.block >= 0;
}

std::size_t PagedCacheManager::num_blocks(std::uintptr_t node_key) const {
    return get_state(node_key).num_blocks;
}

std::size_t PagedCacheManager::block_size(std::uintptr_t node_key) const {
    return get_state(node_key).block_size;
}

std::size_t PagedCacheManager::num_kv_heads(std::uintptr_t node_key) const {
    return get_state(node_key).num_kv_heads;
}

std::size_t PagedCacheManager::key_head_size(std::uintptr_t node_key) const {
    return get_state(node_key).key_head_size;
}

std::size_t PagedCacheManager::value_head_size(std::uintptr_t node_key) const {
    return get_state(node_key).value_head_size;
}

void PagedCacheManager::update_attention_scores(std::uintptr_t node_key,
                                                const float* scores,
                                                std::size_t scores_len,
                                                const std::int32_t* past_lens,
                                                std::size_t seq_count) {
    auto& st = get_state(node_key);
    if (seq_count > st.sequences.size()) {
        seq_count = st.sequences.size();
    }
    const std::int32_t bs = static_cast<std::int32_t>(st.block_size);
    if (bs <= 0 || scores == nullptr || scores_len == 0) {
        return;
    }

    // scores layout: flat concatenation of [past_lens[s] + new_len_s] per sequence
    // we accumulate per-block scores by summing the per-token values
    std::size_t offset = 0;
    for (std::size_t s = 0; s < seq_count; ++s) {
        auto& seq = st.sequences[s];
        const std::size_t total_len = (past_lens && s < seq_count) ? static_cast<std::size_t>(seq.logical_length) : 0;
        if (total_len == 0 || seq.blocks.empty()) {
            offset += total_len;
            continue;
        }

        // make sure block_scores has the right size
        seq.block_scores.resize(seq.blocks.size(), 0.f);

        // accumulate per-token attention scores into per-block sums
        for (std::size_t t = 0; t < total_len && (offset + t) < scores_len; ++t) {
            const std::int32_t token_pos = static_cast<std::int32_t>(t);
            if (token_pos < seq.trim_front) {
                continue;
            }
            const std::int32_t rel = token_pos - seq.trim_front;
            const std::size_t block_idx = static_cast<std::size_t>(rel / bs);
            if (block_idx < seq.block_scores.size()) {
                seq.block_scores[block_idx] += scores[offset + t];
            }
        }
        offset += total_len;
    }
}

void PagedCacheManager::update_diversity_scores(std::uintptr_t node_key,
                                                std::size_t seq_idx,
                                                const float* diversity_matrix,
                                                std::size_t num_blocks_in_zone,
                                                std::size_t eviction_size,
                                                std::size_t start_block_offset) {
    auto& st = get_state(node_key);
    if (seq_idx >= st.sequences.size() || diversity_matrix == nullptr || num_blocks_in_zone == 0 ||
        eviction_size == 0) {
        return;
    }

    auto& seq = st.sequences[seq_idx];
    const std::size_t total = num_blocks_in_zone * eviction_size;
    seq.diversity_matrix.assign(diversity_matrix, diversity_matrix + total);
    seq.diversity_n_blocks = num_blocks_in_zone;
    seq.diversity_evict_size = eviction_size;
    seq.diversity_start_block = start_block_offset;
}

}  // namespace paged_attention_cache
}  // namespace reference
}  // namespace ov
