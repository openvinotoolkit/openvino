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

namespace ov {
namespace internal {

class PagedCacheManager {
public:
    using handle_t = std::uint64_t;

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

    struct page_span {
        std::size_t key_byte_offset{0};
        std::size_t value_byte_offset{0};
        std::size_t byte_length{0};
    };

    struct snapshot_t {
        ov::element::Type elem_type{};
        std::size_t total_bytes{0};
        std::size_t page_bytes{0};
        std::size_t alignment_bytes{0};
        std::size_t num_pages{0};
        std::size_t free_pages{0};
        struct op_info {
            handle_t handle{0};
            std::vector<std::size_t> pages;
            std::vector<float> scores;
            std::vector<std::int32_t> subseq_begins;
            // stored geometry if provided after shape inference:
            std::size_t num_blocks{0}, num_heads{0}, block_size{0};
            std::size_t key_head_size{0}, value_head_size{0}, query_head_size{0};
        };
        std::vector<op_info> ops;
    };

    // ctor/dtor
    PagedCacheManager(ov::element::Type elem_type,
                      std::size_t total_bytes,
                      std::size_t page_bytes,
                      std::size_t alignment_bytes = 64);
    ~PagedCacheManager();

    PagedCacheManager(const PagedCacheManager&) = delete;
    PagedCacheManager& operator=(const PagedCacheManager&) = delete;

    // register/unregister a PagedAttention via any of its outputs
    handle_t register_operator(const ov::Output<ov::Node>& pa_output);
    void unregister_operator(handle_t h);

    // shared buffer access
    cache_blocks get_cache_blocks() const noexcept;
    void* get_key_base() const noexcept;
    void* get_value_base() const noexcept;
    std::size_t get_total_bytes() const noexcept;
    std::size_t get_page_bytes() const noexcept;
    std::size_t get_num_pages() const noexcept;
    ov::element::Type get_element_type() const noexcept;

    // per-operator metadata
    subsequence_view get_subsequence_begins(handle_t h) const;

    // set per-operator cache geometry after PA shape inference
    void set_operator_cache_geometry(handle_t h,
                                     std::size_t num_blocks,
                                     std::size_t num_heads,
                                     std::size_t block_size,
                                     std::size_t key_head_size,
                                     std::size_t value_head_size,
                                     std::size_t query_head_size);

    // largest contiguous span within requested range owned by operator
    page_span get_page_span(handle_t h, std::size_t start_page, std::size_t page_count) const;

    // page lifecycle
    std::vector<std::size_t> acquire_pages(handle_t h, std::size_t page_count);
    void release_pages(handle_t h, const std::vector<std::size_t>& pages);

    // insert and scoring
    template <typename T>
    std::vector<std::size_t> insert(handle_t h,
                                    const T* key_src,
                                    const T* value_src,
                                    std::size_t page_count,
                                    const T* scores /* may be nullptr */);

    template <typename T>
    void set_page_scores(handle_t h, const std::vector<std::size_t>& page_indices, const T* scores);

    // evict to maintain free pool
    void evict_to_target_free(std::size_t target_free_pages);

    // query snapshot for tests/logs
    snapshot_t snapshot() const;

    // tiny debug trace
    static void debug_trace(const char* fmt, ...) noexcept;

private:
    struct page_t {
        std::size_t index{0};
        float score{std::numeric_limits<float>::infinity()};
        handle_t owner{0};
    };

    struct operator_state {
        ov::Output<ov::Node> node_output;
        std::vector<std::size_t> pages;
        std::vector<float> scores;
        std::vector<std::int32_t> subseq_begins;

        // optional geometry set after PA shape inference
        std::size_t num_blocks{0}, num_heads{0}, block_size{0};
        std::size_t key_head_size{0}, value_head_size{0}, query_head_size{0};
    };

    // helpers
    void* offset_key(std::size_t page_idx) const noexcept;
    void* offset_value(std::size_t page_idx) const noexcept;

    void compute_subsequence_begins_unlocked(operator_state& st) const;

    void ensure_free_pages_unlocked(std::size_t need_pages);
    void evict_one_unlocked();

    std::vector<std::size_t> acquire_pages_unlocked(handle_t h, std::size_t page_count);
    void copy_pages_into_buffers_unlocked(const void* key_src_bytes,
                                          const void* value_src_bytes,
                                          const std::vector<std::size_t>& page_idxs);
    void set_scores_for_pages_unlocked(handle_t h, const std::vector<std::size_t>& page_idxs, const float* scores);

    static float cast_score_to_float(ov::element::Type et, const void* src_scalar) noexcept;
    static bool is_element_compatible_with_T(ov::element::Type et, size_t sizeofT) noexcept;

    void rebuild_evict_heap_unlocked();
    static bool heap_less_(const std::vector<page_t>& pages, std::size_t a, std::size_t b) noexcept;

private:
    ov::element::Type m_elem_type{};
    const std::size_t m_total_bytes{0};
    const std::size_t m_page_bytes{0};
    const std::size_t m_alignment_bytes{64};
    const std::size_t m_num_pages{0};

    void* m_key_base{nullptr};
    void* m_value_base{nullptr};

    std::vector<page_t> m_pages;
    std::list<std::size_t> m_free_page_list;
    std::unordered_map<handle_t, operator_state> m_ops;

    std::vector<std::size_t> m_evict_heap;

    std::atomic<handle_t> m_next_handle{1};
    mutable std::mutex m_mu;
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
inline std::size_t PagedCacheManager::get_page_bytes() const noexcept {
    return m_page_bytes;
}
inline std::size_t PagedCacheManager::get_num_pages() const noexcept {
    return m_num_pages;
}
inline ov::element::Type PagedCacheManager::get_element_type() const noexcept {
    return m_elem_type;
}

template <typename T>
std::vector<std::size_t> PagedCacheManager::insert(handle_t h,
                                                   const T* key_src,
                                                   const T* value_src,
                                                   std::size_t page_count,
                                                   const T* scores) {
    if (!is_element_compatible_with_T(m_elem_type, sizeof(T))) {
        throw std::runtime_error("PagedCacheManager::insert<T>: T does not match element type");
    }
    if (page_count == 0)
        return {};

    std::lock_guard<std::mutex> lk(m_mu);

    auto page_idxs = acquire_pages_unlocked(h, page_count);

    const void* kbytes = static_cast<const void*>(key_src);
    const void* vbytes = static_cast<const void*>(value_src);
    copy_pages_into_buffers_unlocked(kbytes, vbytes, page_idxs);

    if (scores) {
        std::vector<float> fs(page_count);
        for (std::size_t i = 0; i < page_count; ++i) {
            fs[i] = cast_score_to_float(m_elem_type, static_cast<const void*>(&scores[i]));
        }
        set_scores_for_pages_unlocked(h, page_idxs, fs.data());
    }

    debug_trace("[CM] insert: h=%llu pages=%zu\n", static_cast<unsigned long long>(h), page_idxs.size());

    return page_idxs;
}

template <typename T>
void PagedCacheManager::set_page_scores(handle_t h, const std::vector<std::size_t>& page_indices, const T* scores) {
    if (!is_element_compatible_with_T(m_elem_type, sizeof(T))) {
        throw std::runtime_error("PagedCacheManager::set_page_scores<T>: T does not match element type");
    }
    if (page_indices.empty())
        return;

    std::lock_guard<std::mutex> lk(m_mu);

    std::vector<float> fs(page_indices.size());
    for (std::size_t i = 0; i < page_indices.size(); ++i) {
        fs[i] = cast_score_to_float(m_elem_type, static_cast<const void*>(&scores[i]));
    }
    set_scores_for_pages_unlocked(h, page_indices, fs.data());

    debug_trace("[CM] set_page_scores: h=%llu count=%zu\n", static_cast<unsigned long long>(h), page_indices.size());
}

}  // namespace internal
}  // namespace ov
