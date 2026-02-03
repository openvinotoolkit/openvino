// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/paged_cache_manager.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "openvino/reference/convert.hpp"

void* ov::reference::paged_attention_cache::PagedCacheManager::get_key_base() const noexcept {
    return get_cache_blocks().key_base;
}
void* ov::reference::paged_attention_cache::PagedCacheManager::get_value_base() const noexcept {
    return get_cache_blocks().value_base;
}
std::size_t ov::reference::paged_attention_cache::PagedCacheManager::get_total_bytes() const noexcept {
    return m_total_bytes;
}
ov::element::Type ov::reference::paged_attention_cache::PagedCacheManager::get_element_type() const noexcept {
    return m_elem_type;
}
std::size_t ov::reference::paged_attention_cache::PagedCacheManager::get_num_blocks() noexcept {
    return m_num_blocks;
}
std::size_t ov::reference::paged_attention_cache::PagedCacheManager::get_block_size() noexcept {
    return m_block_size;
}
std::size_t ov::reference::paged_attention_cache::PagedCacheManager::get_block_bytes() noexcept {
    return m_key_block_bytes;
}

ov::reference::paged_attention_cache::PagedCacheManager::PagedCacheManager(ov::element::Type elem_type,
                                                                           std::size_t total_bytes)
    : m_elem_type(elem_type),
      m_total_bytes(total_bytes) {
    // Check for perfect split of bytes
    if (m_total_bytes % 2) {
        OPENVINO_THROW("PagedCacheManager: total allocated bytes must be divisible by 2");
    }

    const size_t half_bytes = m_total_bytes / 2;

    try {
        m_key_buffer = ov::AlignedBuffer(half_bytes, 64);
        m_value_buffer = ov::AlignedBuffer(half_bytes, 64);
    } catch (const std::bad_alloc&) {
        OPENVINO_THROW("PagedCacheManager: aligned allocation failed");
    }
}

// every op registered once
bool ov::reference::paged_attention_cache::PagedCacheManager::operator_registered(const size_t node_id) {
    return m_ops.count(node_id);
}

// registration
// returns unique node_id id assigned to the given node_id
size_t ov::reference::paged_attention_cache::PagedCacheManager::register_operator(const size_t block_size,
                                                                                  const size_t num_heads,
                                                                                  const size_t key_head_size,
                                                                                  const size_t value_head_size,
                                                                                  const size_t query_head_size) {
    operator_state state;
    state.node_id = m_node_id++;
    compute_operator_cache_geometry(state, block_size, num_heads, key_head_size, value_head_size, query_head_size);
    m_ops.emplace(state.node_id, std::move(state));

    return state.node_id;
}

// buffers
ov::reference::paged_attention_cache::PagedCacheManager::CacheBlocks
ov::reference::paged_attention_cache::PagedCacheManager::get_cache_blocks() const noexcept {
    const std::size_t half_bytes = m_total_bytes / 2;
    return CacheBlocks{const_cast<void*>(m_key_buffer.get_ptr()),
                       const_cast<void*>(m_value_buffer.get_ptr()),
                       half_bytes,
                       half_bytes};
}

// per-operator metadata
ov::reference::paged_attention_cache::PagedCacheManager::SubsequenceView
ov::reference::paged_attention_cache::PagedCacheManager::get_subsequence_begins(size_t node_id) const {
    auto it = m_ops.find(node_id);
    if (it == m_ops.end())
        return {};
    const auto& v = it->second.subsequence_begins;
    return SubsequenceView{v.data(), v.size()};
}

// Check if sizes match, initialize empty blocks since we know the size of a block and dtype at this point, prepare a
// state for a given PA
void ov::reference::paged_attention_cache::PagedCacheManager::compute_operator_cache_geometry(
    operator_state& state,
    const size_t block_size,
    const size_t num_heads,
    const size_t key_head_size,
    const size_t value_head_size,
    const size_t query_head_size) {
    if (m_block_size != block_size) {
        if (!m_block_size) {
            m_block_size = block_size;
        } else {
            OPENVINO_THROW("PagedCacheManager: All PagedAttention nodes must have the same block size.");
        }
    }

    if (m_num_heads != num_heads) {
        if (!m_num_heads) {
            m_num_heads = num_heads;
        } else {
            OPENVINO_THROW("PagedCacheManager: All PagedAttention nodes must have the same number of heads.");
        }
    }

    if (m_key_head_size != key_head_size) {
        if (!m_key_head_size) {
            m_key_head_size = key_head_size;
        } else {
            OPENVINO_THROW("PagedCacheManager: All PagedAttention nodes must have the same number of key cache heads.");
        }
    }

    if (m_value_head_size != value_head_size) {
        if (!m_value_head_size) {
            m_value_head_size = value_head_size;
        } else {
            OPENVINO_THROW(
                "PagedCacheManager: All PagedAttention nodes must have the same number of value cache heads.");
        }
    }

    if (!m_num_blocks) {
        const size_t elem_bytes = static_cast<size_t>(m_elem_type.size());
        m_key_block_bytes = m_num_heads * m_block_size * m_key_head_size * elem_bytes;
        m_value_block_bytes = m_num_heads * m_block_size * m_value_head_size * elem_bytes;

        const size_t half_bytes = m_total_bytes / 2;
        size_t num_blocks_key_cache = (m_key_block_bytes > 0) ? (half_bytes / m_key_block_bytes) : 0;
        size_t num_blocks_value_cache = (m_value_block_bytes > 0) ? (half_bytes / m_value_block_bytes) : 0;
        // If unequal we don't have the same available blocks in each cache.
        // Instead of reordering memory, for now it's simpler to just set max blocks count to lower bound.
        m_num_blocks = std::min(num_blocks_key_cache, num_blocks_value_cache);

        m_blocks.resize(m_num_blocks);
        for (std::size_t i = 0; i < m_num_blocks; ++i) {
            m_blocks[i].index = i;
            m_blocks[i].score = std::numeric_limits<float>::infinity();
            m_blocks[i].owner = 0;
            m_free_block_list.push_back(i);
        }
        rebuild_evict_heap_unlocked();
    }

    state.num_blocks = 0;
    state.num_heads = num_heads;
    state.block_size = block_size;
    state.key_head_size = key_head_size;
    state.value_head_size = value_head_size;
    state.query_head_size = query_head_size;  // unused?
}

// block mgmt
std::vector<std::size_t> ov::reference::paged_attention_cache::PagedCacheManager::acquire_blocks(
    size_t node_id,
    std::size_t block_count) {
    return acquire_blocks_unlocked(node_id, block_count);
}

void ov::reference::paged_attention_cache::PagedCacheManager::release_blocks(size_t node_id,
                                                                             const std::vector<std::size_t>& blocks) {
    auto it = m_ops.find(node_id);
    if (it == m_ops.end())
        return;
    auto& state = it->second;

    for (std::size_t block_id : blocks) {
        if (block_id >= m_blocks.size())
            continue;
        if (m_blocks[block_id].owner != node_id)
            continue;

        auto pit = std::find(state.blocks.begin(), state.blocks.end(), block_id);
        if (pit != state.blocks.end()) {
            const auto pos = static_cast<std::size_t>(std::distance(state.blocks.begin(), pit));
            state.blocks.erase(pit);
            if (pos < state.scores.size())
                state.scores.erase(state.scores.begin() + static_cast<std::ptrdiff_t>(pos));
        }

        m_blocks[block_id].owner = 0;
        m_blocks[block_id].score = std::numeric_limits<float>::infinity();
        m_free_block_list.push_back(block_id);
    }
    rebuild_evict_heap_unlocked();
}

// insert & scoring helpers
std::vector<std::size_t> ov::reference::paged_attention_cache::PagedCacheManager::acquire_blocks_unlocked(
    size_t node_id,
    std::size_t block_count) {
    if (block_count == 0)
        return {};
    auto it = m_ops.find(node_id);
    if (it == m_ops.end())
        OPENVINO_THROW("PagedCacheManager::acquire_blocks_unlocked: unknown node_id (not registered)");
    auto& state = it->second;

    ensure_free_blocks_unlocked(block_count);

    std::vector<std::size_t> granted;
    granted.reserve(block_count);
    for (std::size_t i = 0; i < block_count; ++i) {
        if (m_free_block_list.empty()) {
            evict_one_unlocked();
        }
        if (m_free_block_list.empty())
            break;

        const std::size_t block_id = m_free_block_list.front();
        m_free_block_list.pop_front();

        m_blocks[block_id].owner = node_id;
        m_blocks[block_id].score = std::numeric_limits<float>::infinity();

        state.blocks.push_back(block_id);
        state.scores.push_back(std::numeric_limits<float>::infinity());
        granted.push_back(block_id);
    }

    rebuild_evict_heap_unlocked();
    return granted;
}

void ov::reference::paged_attention_cache::PagedCacheManager::copy_blocks_into_buffers_unlocked(
    const void* key_src_bytes,
    const void* value_src_bytes,
    const std::vector<std::size_t>& block_idxs) {
    if (block_idxs.empty())
        return;
    const auto* ksrc = static_cast<const unsigned char*>(key_src_bytes);
    const auto* vsrc = static_cast<const unsigned char*>(value_src_bytes);
    const std::size_t key_bytes_per_block = m_key_block_bytes;
    const std::size_t value_bytes_per_block = m_value_block_bytes;

    for (std::size_t i = 0; i < block_idxs.size(); ++i) {
        const std::size_t p = block_idxs[i];
        std::memcpy(static_cast<unsigned char*>(offset_key(p)), ksrc + i * key_bytes_per_block, key_bytes_per_block);
        std::memcpy(static_cast<unsigned char*>(offset_value(p)),
                    vsrc + i * value_bytes_per_block,
                    value_bytes_per_block);
    }
}

void ov::reference::paged_attention_cache::PagedCacheManager::set_scores_for_blocks_unlocked(
    size_t node_id,
    const std::vector<std::size_t>& block_idxs,
    const float* scores) {
    auto it = m_ops.find(node_id);
    if (it == m_ops.end())
        OPENVINO_THROW("PagedCacheManager::set_scores_for_blocks_unlocked: unknown handle");
    auto& state = it->second;

    for (std::size_t i = 0; i < block_idxs.size(); ++i) {
        const std::size_t block_id = block_idxs[i];
        if (block_id >= m_blocks.size())
            continue;
        if (m_blocks[block_id].owner != node_id)
            continue;

        auto pit = std::find(state.blocks.begin(), state.blocks.end(), block_id);
        if (pit != state.blocks.end()) {
            const auto pos = static_cast<std::size_t>(std::distance(state.blocks.begin(), pit));
            if (pos < state.scores.size())
                state.scores[pos] = scores[i];
        }
        m_blocks[block_id].score = scores[i];
    }
    rebuild_evict_heap_unlocked();
}

// eviction
void ov::reference::paged_attention_cache::PagedCacheManager::ensure_free_blocks_unlocked(std::size_t need_blocks) {
    if (m_free_block_list.size() >= need_blocks)
        return;
    const std::size_t deficit = need_blocks - m_free_block_list.size();
    for (std::size_t i = 0; i < deficit; ++i) {
        evict_one_unlocked();
        if (m_free_block_list.size() >= need_blocks)
            break;
    }
}

void ov::reference::paged_attention_cache::PagedCacheManager::evict_one_unlocked() {
    if (m_evict_heap.empty())
        rebuild_evict_heap_unlocked();
    if (m_evict_heap.empty())
        return;

    std::pop_heap(m_evict_heap.begin(), m_evict_heap.end(), [this](std::size_t a, std::size_t b) {
        return heap_less(m_blocks, a, b);
    });
    const std::size_t victim = m_evict_heap.back();
    m_evict_heap.pop_back();

    if (victim >= m_blocks.size())
        return;
    auto& pg = m_blocks[victim];
    if (pg.owner == 0)
        return;

    const size_t owner = pg.owner;
    auto it = m_ops.find(owner);
    if (it != m_ops.end()) {
        auto& state = it->second;
        auto pit = std::find(state.blocks.begin(), state.blocks.end(), victim);
        if (pit != state.blocks.end()) {
            const auto pos = static_cast<std::size_t>(std::distance(state.blocks.begin(), pit));
            state.blocks.erase(pit);
            if (pos < state.scores.size())
                state.scores.erase(state.scores.begin() + static_cast<std::ptrdiff_t>(pos));
        }
    }

    pg.owner = 0;
    pg.score = std::numeric_limits<float>::infinity();
    m_free_block_list.push_back(victim);
}

void ov::reference::paged_attention_cache::PagedCacheManager::evict_to_target_free(std::size_t target_free_blocks) {
    while (m_free_block_list.size() < target_free_blocks) {
        evict_one_unlocked();
        if (m_evict_heap.empty())
            break;
    }
}

void* ov::reference::paged_attention_cache::PagedCacheManager::offset_key(std::size_t block_idx) const noexcept {
    unsigned char* base = static_cast<unsigned char*>(const_cast<void*>(m_key_buffer.get_ptr()));
    return static_cast<void*>(base + block_idx * m_key_block_bytes);
}

void* ov::reference::paged_attention_cache::PagedCacheManager::offset_value(std::size_t block_idx) const noexcept {
    unsigned char* base = static_cast<unsigned char*>(const_cast<void*>(m_value_buffer.get_ptr()));
    return static_cast<void*>(base + block_idx * m_value_block_bytes);
}

float ov::reference::paged_attention_cache::PagedCacheManager::cast_score_to_float(ov::element::Type et,
                                                                                   const void* src_scalar) noexcept {
    float dst = 0.0f;
    switch (et) {
    case ov::element::f32:
        ov::reference::convert(static_cast<const float*>(src_scalar), &dst, 1);
        break;

    case ov::element::f16:
        ov::reference::convert(static_cast<const ov::float16*>(src_scalar), &dst, 1);
        break;

    case ov::element::bf16:
        ov::reference::convert(static_cast<const ov::bfloat16*>(src_scalar), &dst, 1);
        break;

    case ov::element::i32:
        ov::reference::convert(static_cast<const int32_t*>(src_scalar), &dst, 1);
        break;

    case ov::element::i64:
        ov::reference::convert(static_cast<const int64_t*>(src_scalar), &dst, 1);
        break;

    case ov::element::u32:
        ov::reference::convert(static_cast<const uint32_t*>(src_scalar), &dst, 1);
        break;

    case ov::element::u64:
        ov::reference::convert(static_cast<const uint64_t*>(src_scalar), &dst, 1);
        break;

    default:
        ov::reference::convert(static_cast<const float*>(src_scalar), &dst, 1);
        break;
    }
    return dst;
}

bool ov::reference::paged_attention_cache::PagedCacheManager::is_element_compatible_with_T(ov::element::Type et,
                                                                                           size_t sizeofT) noexcept {
    return sizeofT == et.size();
}

void ov::reference::paged_attention_cache::PagedCacheManager::rebuild_evict_heap_unlocked() {
    m_evict_heap.clear();
    m_evict_heap.reserve(m_blocks.size());
    for (const auto& pg : m_blocks) {
        if (pg.owner != 0 && std::isfinite(pg.score)) {
            m_evict_heap.push_back(pg.index);
        }
    }
    std::make_heap(m_evict_heap.begin(), m_evict_heap.end(), [this](std::size_t a, std::size_t b) {
        return heap_less(m_blocks, a, b);
    });
}

bool ov::reference::paged_attention_cache::PagedCacheManager::heap_less(const std::vector<block_t>& blocks,
                                                                        std::size_t a,
                                                                        std::size_t b) noexcept {
    const float sa = (a < blocks.size()) ? blocks[a].score : std::numeric_limits<float>::infinity();
    const float sb = (b < blocks.size()) ? blocks[b].score : std::numeric_limits<float>::infinity();
    return sa > sb;  // lower score = higher eviction priority
}
