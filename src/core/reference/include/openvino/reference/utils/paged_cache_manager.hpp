// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace reference {
namespace paged_attention_cache {

// A single-threaded CacheManager used by the reference implementation of PagedAttentionExtension.
class PagedCacheManager {
public:
    struct TokenAddress {
        std::int32_t block = -1;
        std::int32_t offset = 0;
    };

    explicit PagedCacheManager(ov::element::Type elem_type);
    PagedCacheManager(const PagedCacheManager&) = delete;
    PagedCacheManager& operator=(const PagedCacheManager&) = delete;

    // Register (or find) an operator state for a PagedAttentionExtension node
    bool ensure_operator(std::uintptr_t node_key,
                         const void* key_cache_init,
                         const void* value_cache_init,
                         const ov::Shape& key_cache_shape,
                         const ov::Shape& value_cache_shape,
                         const std::int32_t* block_indices_init,
                         std::size_t block_indices_count,
                         const std::int32_t* block_indices_begins_init,
                         std::size_t block_indices_begins_count,
                         const std::int32_t* past_lens_init,
                         std::size_t past_lens_count);

    // Update per-sequence logical lengths at the beginning of a step
    //
    // The reference implementation assumes that tokens are appended and that past_lens represents the current
    // logical length of each sequence before appending the new tokens from this call.
    void begin_step(std::uintptr_t node_key, const std::int32_t* past_lens, std::size_t seq_count);

    // Ensure storage for a token position within a sequence
    //
    // token_pos is the logical token index within the sequence timeline (0...logical_length - 1)
    // This function allocates (or reuses) blocks as needed
    TokenAddress ensure_token(std::uintptr_t node_key, std::size_t seq_idx, std::int32_t token_pos);

    /// Resolve an existing token address
    /// Returns false if the token is out of the retained window (trimmed) or beyond logical length
    bool resolve_token(std::uintptr_t node_key,
                       std::size_t seq_idx,
                       std::int32_t token_pos,
                       TokenAddress& out_addr) const;

    // Write key/value vectors for a token into the cache
    //
    // key_row layout:   [num_kv_heads * key_head_size]
    // value_row layout: [num_kv_heads * value_head_size]
    template <typename T>
    void write_token_kv(std::uintptr_t node_key,
                        std::size_t seq_idx,
                        std::int32_t token_pos,
                        const T* key_row,
                        const T* value_row);

    // Access pointers for an already resolved token
    template <typename T>
    const T* key_ptr(std::uintptr_t node_key, TokenAddress addr, std::size_t kv_head) const;

    template <typename T>
    const T* value_ptr(std::uintptr_t node_key, TokenAddress addr, std::size_t kv_head) const;

    // Cache layout getters (per node)
    std::size_t num_blocks(std::uintptr_t node_key) const;
    std::size_t block_size(std::uintptr_t node_key) const;
    std::size_t num_kv_heads(std::uintptr_t node_key) const;
    std::size_t key_head_size(std::uintptr_t node_key) const;
    std::size_t value_head_size(std::uintptr_t node_key) const;

    ov::element::Type element_type() const noexcept {
        return m_elem_type;
    }

private:
    struct SequenceState {
        std::int32_t logical_length = 0;  // expected by external past_lens
        std::int32_t trim_front = 0;      // number of tokens trimmed from front (multiple of block_size)
        std::deque<std::int32_t> blocks;  // physical block IDs for [trim_front, trim_front + blocks*block_size)
    };

    struct OperatorState {
        // layout
        std::size_t num_blocks = 0;
        std::size_t block_size = 0;
        std::size_t num_kv_heads = 0;
        std::size_t key_head_size = 0;
        std::size_t value_head_size = 0;

        // bytes per block
        std::size_t key_block_bytes = 0;
        std::size_t value_block_bytes = 0;

        // owned storage (copied once from init tensors)
        ov::AlignedBuffer key_cache;
        ov::AlignedBuffer value_cache;

        // free list and per-seq state
        std::vector<std::uint8_t> block_used;   // 0/1
        std::vector<std::int32_t> free_blocks;  // stack
        std::vector<SequenceState> sequences;
    };

    OperatorState& get_state(std::uintptr_t node_key);
    const OperatorState& get_state(std::uintptr_t node_key) const;

    static std::size_t tensor_byte_size(const ov::Shape& shape, std::size_t elem_bytes);

    void init_sequences_from_block_tables(OperatorState& st,
                                          const std::int32_t* block_indices,
                                          std::size_t block_indices_count,
                                          const std::int32_t* block_indices_begins,
                                          std::size_t begins_count,
                                          const std::int32_t* past_lens,
                                          std::size_t past_lens_count);

    std::int32_t allocate_block(OperatorState& st, std::size_t requester_seq);
    std::int32_t steal_block_from_victim(OperatorState& st, std::size_t requester_seq);

    static std::size_t elem_bytes_or_throw(ov::element::Type et);

    static void validate_cache_rank4_or_throw(const ov::Shape& key_cache_shape, const ov::Shape& value_cache_shape);

    static void parse_cache_layout_or_throw(OperatorState& st,
                                            const ov::Shape& key_cache_shape,
                                            const ov::Shape& value_cache_shape,
                                            std::size_t elem_bytes);

    template <typename T>
    static void memcpy_typed(void* dst, const void* src, std::size_t count) {
        std::memcpy(dst, src, count * sizeof(T));
    }

    // compute base pointers into internal storage
    template <typename T>
    static T* key_block_base(OperatorState& st, std::int32_t block_id, std::size_t kv_head) {
        auto* base = static_cast<T*>(st.key_cache.get_ptr());
        const std::size_t block_stride = st.num_kv_heads * st.block_size * st.key_head_size;
        const std::size_t kv_stride = st.block_size * st.key_head_size;
        return base + static_cast<std::size_t>(block_id) * block_stride + kv_head * kv_stride;
    }

    template <typename T>
    static T* value_block_base(OperatorState& st, std::int32_t block_id, std::size_t kv_head) {
        auto* base = static_cast<T*>(st.value_cache.get_ptr());
        const std::size_t block_stride = st.num_kv_heads * st.block_size * st.value_head_size;
        const std::size_t kv_stride = st.block_size * st.value_head_size;
        return base + static_cast<std::size_t>(block_id) * block_stride + kv_head * kv_stride;
    }

    template <typename T>
    static const T* key_block_base(const OperatorState& st, std::int32_t block_id, std::size_t kv_head) {
        auto* base = static_cast<const T*>(st.key_cache.get_ptr());
        const std::size_t block_stride = st.num_kv_heads * st.block_size * st.key_head_size;
        const std::size_t kv_stride = st.block_size * st.key_head_size;
        return base + static_cast<std::size_t>(block_id) * block_stride + kv_head * kv_stride;
    }

    template <typename T>
    static const T* value_block_base(const OperatorState& st, std::int32_t block_id, std::size_t kv_head) {
        auto* base = static_cast<const T*>(st.value_cache.get_ptr());
        const std::size_t block_stride = st.num_kv_heads * st.block_size * st.value_head_size;
        const std::size_t kv_stride = st.block_size * st.value_head_size;
        return base + static_cast<std::size_t>(block_id) * block_stride + kv_head * kv_stride;
    }

private:
    ov::element::Type m_elem_type;
    std::unordered_map<std::uintptr_t, OperatorState> m_ops;
};

// ---------------- template impl ----------------

template <typename T>
void PagedCacheManager::write_token_kv(std::uintptr_t node_key,
                                       std::size_t seq_idx,
                                       std::int32_t token_pos,
                                       const T* key_row,
                                       const T* value_row) {
    auto& st = get_state(node_key);
    if (seq_idx >= st.sequences.size()) {
        OPENVINO_THROW("PagedCacheManager::write_token_kv: seq_idx out of range");
    }

    const auto addr = ensure_token(node_key, seq_idx, token_pos);
    if (addr.block < 0) {
        OPENVINO_THROW("PagedCacheManager::write_token_kv: failed to allocate block");
    }

    // Copy per-kv-head vectors into the block storage.
    for (std::size_t kvh = 0; kvh < st.num_kv_heads; ++kvh) {
        T* kdst = key_block_base<T>(st, addr.block, kvh) + static_cast<std::size_t>(addr.offset) * st.key_head_size;
        const T* ksrc = key_row + kvh * st.key_head_size;
        std::memcpy(kdst, ksrc, st.key_head_size * sizeof(T));

        T* vdst = value_block_base<T>(st, addr.block, kvh) + static_cast<std::size_t>(addr.offset) * st.value_head_size;
        const T* vsrc = value_row + kvh * st.value_head_size;
        std::memcpy(vdst, vsrc, st.value_head_size * sizeof(T));
    }

    // Maintains logical length
    auto& seq = st.sequences[seq_idx];
    if (token_pos >= seq.logical_length) {
        seq.logical_length = token_pos + 1;
    }
}

template <typename T>
const T* PagedCacheManager::key_ptr(std::uintptr_t node_key, TokenAddress addr, std::size_t kv_head) const {
    const auto& st = get_state(node_key);
    if (addr.block < 0) {
        return nullptr;
    }
    if (kv_head >= st.num_kv_heads) {
        return nullptr;
    }
    const T* base = key_block_base<T>(st, addr.block, kv_head);
    return base + static_cast<std::size_t>(addr.offset) * st.key_head_size;
}

template <typename T>
const T* PagedCacheManager::value_ptr(std::uintptr_t node_key, TokenAddress addr, std::size_t kv_head) const {
    const auto& st = get_state(node_key);
    if (addr.block < 0) {
        return nullptr;
    }
    if (kv_head >= st.num_kv_heads) {
        return nullptr;
    }
    const T* base = value_block_base<T>(st, addr.block, kv_head);
    return base + static_cast<std::size_t>(addr.offset) * st.value_head_size;
}

}  // namespace paged_attention_cache
}  // namespace reference
}  // namespace ov
