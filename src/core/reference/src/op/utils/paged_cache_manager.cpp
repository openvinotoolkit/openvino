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

PagedCacheManager::PagedCacheManager(ov::element::Type elem_type) : m_elem_type(elem_type) {}

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

    // If no block tables were provided, initialize empty sequences.
    if (block_indices == nullptr || block_indices_begins == nullptr || begins_count != (seq_count + 1)) {
        for (std::size_t s = 0; s < seq_count; ++s) {
            st.sequences[s].logical_length = past_lens ? past_lens[s] : 0;
        }
        return;
    }

    // Fill per-sequence block lists based on provided tables.
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

        // Ensure trim_front is block-aligned and within logical length.
        seq.trim_front = 0;
    }
}

std::int32_t PagedCacheManager::steal_block_from_victim(OperatorState& st, std::size_t requester_seq) {
    // Choose a victim sequence with the most retained blocks. Deterministic tie-breaking by index.
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
        // As a last resort, steal from the requester itself
        best_seq = requester_seq;
    }

    auto& victim = st.sequences[best_seq];
    if (victim.blocks.empty()) {
        OPENVINO_THROW("PagedCacheManager: cannot steal a block (no victim blocks)");
    }

    const std::int32_t bid = victim.blocks.front();
    victim.blocks.pop_front();
    victim.trim_front += static_cast<std::int32_t>(st.block_size);
    return bid;
}

std::int32_t PagedCacheManager::allocate_block(OperatorState& st, std::size_t requester_seq) {
    if (!st.free_blocks.empty()) {
        const std::int32_t bid = st.free_blocks.back();
        st.free_blocks.pop_back();
        if (bid >= 0 && static_cast<std::size_t>(bid) < st.block_used.size()) {
            st.block_used[static_cast<std::size_t>(bid)] = 1;
        }
        return bid;
    }
    // No free blocks left: evict one block from a victim sequence.
    return steal_block_from_victim(st, requester_seq);
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

    // Initialize free list and per-seq state.
    st.block_used.assign(st.num_blocks, 0);

    init_sequences_from_block_tables(st,
                                     block_indices_init,
                                     block_indices_count,
                                     block_indices_begins_init,
                                     block_indices_begins_count,
                                     past_lens_init,
                                     past_lens_count);

    // Build free block stack.
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
            // External timeline reset/truncation: drop cached blocks for this sequence
            // In the reference implementation, blocks are not shared between sequences, so we can
            // safely return them to the global free list
            for (const std::int32_t bid : seq.blocks) {
                if (bid >= 0 && static_cast<std::size_t>(bid) < st.num_blocks && st.block_used[bid]) {
                    st.block_used[bid] = 0;
                    st.free_blocks.push_back(bid);
                }
            }
            seq.blocks.clear();
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

    const std::int32_t rel = token_pos - seq.trim_front;
    const std::int32_t bs = static_cast<std::int32_t>(st.block_size);
    const std::int32_t block_index = (bs > 0) ? (rel / bs) : 0;
    const std::int32_t off = (bs > 0) ? (rel % bs) : 0;

    while (seq.blocks.size() <= static_cast<std::size_t>(block_index)) {
        const std::int32_t bid = allocate_block(st, seq_idx);
        seq.blocks.push_back(bid);
    }

    // If we exceeded the global capacity (all blocks used), allocate_block will have trimmed a victim
    // Also ensure the requester sequence doesn't exceed its representable window

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

}  // namespace paged_attention_cache
}  // namespace reference
}  // namespace ov
