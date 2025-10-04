#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

#ifndef CM_DEBUG
#    define CM_DEBUG 0
#endif

// The following value defines the default cache size of a full cache.
// There is so far no other method to set it (constructor is provided, but there is no way for user to input it yet).
//
// The number of elements that fit inside is based on the element type of the cache.
// That means, since PA needs a key and a value cache, each cache receives half the bytes.
// This cache size does not include the "utilities" such as vector of available blocks, block indices, ect.
#define CACHE_SIZE 1000000

namespace ov {
namespace internal {

class PagedCacheManager {
public:
    struct cache_blocks {
        void* key_base{nullptr};
        void* value_base{nullptr};
        size_t key_bytes{0};
        size_t value_bytes{0};
    };

    struct subsequence_view {
        const std::int32_t* data{nullptr};
        size_t count{0};
    };

    struct block_span {
        std::size_t key_byte_offset{0};
        std::size_t value_byte_offset{0};
        std::size_t byte_length{0};
    };

    PagedCacheManager(ov::element::Type elem_type, std::size_t total_bytes = CACHE_SIZE);
    ~PagedCacheManager();

    PagedCacheManager(const PagedCacheManager&) = delete;
    PagedCacheManager& operator=(const PagedCacheManager&) = delete;

    // adds a PagedAttention to the pool of managed ops if not added before
    void register_operator(const std::shared_ptr<ov::Node> node);
    bool operator_registered(const std::shared_ptr<ov::Node> node);

    // shared buffer access
    cache_blocks get_cache_blocks() const noexcept;
    void* get_key_base() const noexcept;
    void* get_value_base() const noexcept;
    std::size_t get_total_bytes() const noexcept;
    ov::element::Type get_element_type() const noexcept;
    std::size_t get_num_blocks() noexcept;
    std::size_t get_block_size() noexcept;
    std::size_t get_block_bytes() noexcept;

    // per-operator metadata
    subsequence_view get_subsequence_begins(std::shared_ptr<ov::Node> node) const;

    // block lifecycle
    std::vector<std::size_t> acquire_blocks(std::shared_ptr<ov::Node> node, std::size_t block_count);
    void release_blocks(std::shared_ptr<ov::Node> node, const std::vector<std::size_t>& blocks);

    // insert and scoring
    template <typename T>
    std::vector<std::size_t> insert(std::shared_ptr<ov::Node> node,
                                    const T* key_src,
                                    const T* value_src,
                                    std::size_t block_count,
                                    const T* scores /* may be nullptr */);

    template <typename T>
    void set_block_scores(std::shared_ptr<ov::Node> node,
                          const std::vector<std::size_t>& block_indices,
                          const T* scores);

    // evict to maintain free pool
    void evict_to_target_free(std::size_t target_free_blocks);

private:
    struct block_t {
        std::size_t index{0};
        float score{std::numeric_limits<float>::infinity()};
        std::shared_ptr<ov::Node> owner{0};
    };

    struct operator_state {
        std::shared_ptr<ov::Node> node;
        std::vector<std::size_t> blocks;
        std::vector<float> scores;
        std::vector<std::int32_t> subsequence_begins;

        std::size_t num_blocks{0}, num_heads{0}, block_size{0};
        std::size_t key_head_size{0}, value_head_size{0}, query_head_size{0};
    };

    // helpers
    void* offset_key(std::size_t block_idx) const noexcept;
    void* offset_value(std::size_t block_idx) const noexcept;

    void compute_subsequence_begins_unlocked(operator_state& st) const;
    void compute_operator_cache_geometry(operator_state& node);

    void ensure_free_blocks_unlocked(std::size_t need_blocks);
    void evict_one_unlocked();

    std::vector<std::size_t> acquire_blocks_unlocked(std::shared_ptr<ov::Node> node, std::size_t block_count);
    void copy_blocks_into_buffers_unlocked(const void* key_src_bytes,
                                           const void* value_src_bytes,
                                           const std::vector<std::size_t>& block_idxs);
    void set_scores_for_blocks_unlocked(std::shared_ptr<ov::Node> node,
                                        const std::vector<std::size_t>& block_idxs,
                                        const float* scores);

    static float cast_score_to_float(ov::element::Type et, const void* src_scalar) noexcept;
    static bool is_element_compatible_with_T(ov::element::Type et, size_t sizeofT) noexcept;

    void rebuild_evict_heap_unlocked();
    static bool heap_less_(const std::vector<block_t>& blocks, std::size_t a, std::size_t b) noexcept;

private:
    const ov::element::Type m_elem_type{};
    const std::size_t m_total_bytes{0};

    std::size_t m_block_size{0};
    std::size_t m_block_bytes{0};  // m_block_size * elem_type_size
    std::size_t m_num_blocks{0};

    void* m_key_base{nullptr};
    void* m_value_base{nullptr};

    std::vector<block_t> m_blocks;
    std::list<std::size_t> m_free_block_list;
    std::unordered_map<std::shared_ptr<ov::Node>, operator_state> m_ops;

    std::vector<std::size_t> m_evict_heap;

    mutable std::mutex m_mutex;
};

// -------- inline trivials & templates --------

inline void* PagedCacheManager::get_key_base() const noexcept {
    return get_cache_blocks().key_base;
}
inline void* PagedCacheManager::get_value_base() const noexcept {
    return get_cache_blocks().value_base;
}
inline std::size_t PagedCacheManager::get_total_bytes() const noexcept {
    return m_total_bytes;
}
inline ov::element::Type PagedCacheManager::get_element_type() const noexcept {
    return m_elem_type;
}
inline std::size_t PagedCacheManager::get_num_blocks() noexcept {
    return m_num_blocks;
}
inline std::size_t PagedCacheManager::get_block_size() noexcept {
    return m_block_size;
}
inline std::size_t PagedCacheManager::get_block_bytes() noexcept {
    return m_block_bytes;
}

template <typename T>
std::vector<std::size_t> PagedCacheManager::insert(std::shared_ptr<ov::Node> node,
                                                   const T* key_src,
                                                   const T* value_src,
                                                   std::size_t block_count,
                                                   const T* scores) {
    if (!is_element_compatible_with_T(m_elem_type, sizeof(T))) {
        throw std::runtime_error("PagedCacheManager::insert<T>: T does not match element type");
    }
    if (block_count == 0)
        return {};

    std::lock_guard<std::mutex> lock(m_mutex);

    auto block_idxs = acquire_blocks_unlocked(node, block_count);

    const void* kbytes = static_cast<const void*>(key_src);
    const void* vbytes = static_cast<const void*>(value_src);
    copy_blocks_into_buffers_unlocked(kbytes, vbytes, block_idxs);

    if (scores) {
        std::vector<float> fs(block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
            fs[i] = cast_score_to_float(m_elem_type, static_cast<const void*>(&scores[i]));
        }
        set_scores_for_blocks_unlocked(node, block_idxs, fs.data());
    }

    return block_idxs;
}

template <typename T>
void PagedCacheManager::set_block_scores(std::shared_ptr<ov::Node> node,
                                         const std::vector<std::size_t>& block_indices,
                                         const T* scores) {
    if (!is_element_compatible_with_T(m_elem_type, sizeof(T))) {
        throw std::runtime_error("PagedCacheManager::set_block_scores<T>: T does not match element type");
    }
    if (block_indices.empty())
        return;

    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<float> fs(block_indices.size());
    for (std::size_t i = 0; i < block_indices.size(); ++i) {
        fs[i] = cast_score_to_float(m_elem_type, static_cast<const void*>(&scores[i]));
    }
    set_scores_for_blocks_unlocked(node, block_indices, fs.data());
}

}  // namespace internal
}  // namespace ov
