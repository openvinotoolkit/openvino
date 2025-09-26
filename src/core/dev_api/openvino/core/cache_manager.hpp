#pragma once
//
// CacheManager.hpp — simple shared paged K/V cache for PagedAttention (reference)
// C++17, conservative toolchains, single global mutex (no shared_mutex)
//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <atomic>
#include <limits>
#include <type_traits>
#include <cstring>   // std::memcpy

#include "openvino/core/node.hpp"   // ov::Node, ov::Output
#include "openvino/core/type/element_type.hpp"

// Set to 1 for very basic logs to stderr.
#ifndef CM_DEBUG
#define CM_DEBUG 0
#endif

namespace ov {
namespace internal {

class CacheManager {
public:
    using Handle = std::uint64_t;

    struct CacheBlocks {
        void*  key_base{nullptr};
        void*  value_base{nullptr};
        size_t key_bytes{0};
        size_t value_bytes{0};
    };

    struct SubsequenceView {
        const std::int32_t* data{nullptr};
        size_t              count{0};
    };

    struct PageSpan {
        std::size_t key_byte_offset{0};
        std::size_t value_byte_offset{0};
        std::size_t byte_length{0};
    };

    struct Snapshot {
        ov::element::Type elem_type{};
        std::size_t total_bytes{0};
        std::size_t page_bytes{0};
        std::size_t alignment_bytes{0};
        std::size_t num_pages{0};
        std::size_t free_pages{0};
        struct OpInfo {
            Handle handle{0};
            std::vector<std::size_t> pages;
            std::vector<float>       scores;
            std::vector<std::int32_t> subseq_begins;
        };
        std::vector<OpInfo> ops;
    };

    CacheManager(ov::element::Type elem_type,
                 std::size_t       total_bytes,
                 std::size_t       page_bytes,
                 std::size_t       alignment_bytes = 64);

    ~CacheManager();

    CacheManager(const CacheManager&) = delete;
    CacheManager& operator=(const CacheManager&) = delete;

    // Register a PagedAttention operator via any of its outputs.
    Handle register_operator(const ov::Output<ov::Node>& pa_output);

    // Unregister and release all pages owned by this operator.
    void unregister_operator(Handle h);

    // Shared buffer access (same for all operators).
    CacheBlocks get_cache_blocks() const noexcept;
    void*       get_key_base()   const noexcept;
    void*       get_value_base() const noexcept;
    std::size_t get_total_bytes() const noexcept;
    std::size_t get_page_bytes()  const noexcept;
    std::size_t get_num_pages()   const noexcept;
    ov::element::Type get_element_type() const noexcept;

    // Per-operator metadata.
    SubsequenceView get_subsequence_begins(Handle h) const;

    // Largest contiguous span within requested range owned by operator.
    PageSpan get_page_span(Handle h, std::size_t start_page, std::size_t page_count) const;

    // Page lifecycle.
    std::vector<std::size_t> acquire_pages(Handle h, std::size_t page_count);
    void release_pages(Handle h, const std::vector<std::size_t>& pages);

    // Insert and scoring.
    template <typename T>
    std::vector<std::size_t> insert(Handle h,
                                    const T* key_src,
                                    const T* value_src,
                                    std::size_t page_count,
                                    const T* scores /* may be nullptr */);

    template <typename T>
    void set_page_scores(Handle h,
                         const std::vector<std::size_t>& page_indices,
                         const T* scores);

    void evict_to_target_free(std::size_t target_free_pages);

    // Introspection (for tests / logging).
    Snapshot snapshot() const;

    // Very small debug helper.
    static void debug_trace(const char* fmt, ...) noexcept;

private:
    struct Page {
        std::size_t index{0};
        float       score{std::numeric_limits<float>::infinity()}; // +inf => not evictable yet
        Handle      owner{0}; // 0 => free
    };

    struct OperatorState {
        ov::Output<ov::Node> node_output; // holds the PA node
        std::vector<std::size_t>   pages;   // owned page indices
        std::vector<float>         scores;  // per-owned-page score (same order as 'pages')
        std::vector<std::int32_t>  subseq_begins; // derived per operator (optional)
    };

    // Helpers implemented in .cpp
    void* offset_key(std::size_t page_idx)   const noexcept;
    void* offset_value(std::size_t page_idx) const noexcept;

    void compute_subsequence_begins_unlocked(OperatorState& st) const;

    void ensure_free_pages_unlocked(std::size_t need_pages);
    void evict_one_unlocked();

    std::vector<std::size_t> acquire_pages_unlocked(Handle h, std::size_t page_count);
    void copy_pages_into_buffers_unlocked(const void* key_src_bytes,
                                          const void* value_src_bytes,
                                          const std::vector<std::size_t>& page_idxs);
    void set_scores_for_pages_unlocked(Handle h,
                                       const std::vector<std::size_t>& page_idxs,
                                       const float* scores);

    static float cast_score_to_float(ov::element::Type et, const void* src_scalar) noexcept;
    static bool  is_element_compatible_with_T(ov::element::Type et, size_t sizeofT) noexcept;

    void rebuild_evict_heap_unlocked();
    static bool heap_less_(const std::vector<Page>& pages, std::size_t a, std::size_t b) noexcept;

private:
    ov::element::Type m_elem_type{};
    const std::size_t m_total_bytes{0};
    const std::size_t m_page_bytes{0};
    const std::size_t m_alignment_bytes{64};
    const std::size_t m_num_pages{0};

    void* m_key_base{nullptr};
    void* m_value_base{nullptr};

    std::vector<Page> m_pages;                 // size = m_num_pages
    std::list<std::size_t> m_free_page_list;   // free page indices
    std::unordered_map<Handle, OperatorState> m_ops;

    // Min-heap (by score) of allocated pages: stores indices into m_pages.
    std::vector<std::size_t> m_evict_heap;

    std::atomic<Handle> m_next_handle{1};

    // Simple coarse-grained mutex (no shared_mutex).
    mutable std::mutex m_mu;
};

// ---------------------- Inline trivials & templates ----------------------

inline void* CacheManager::get_key_base()   const noexcept { return get_cache_blocks().key_base; }
inline void* CacheManager::get_value_base() const noexcept { return get_cache_blocks().value_base; }
inline std::size_t CacheManager::get_total_bytes() const noexcept { return m_total_bytes; }
inline std::size_t CacheManager::get_page_bytes()  const noexcept { return m_page_bytes; }
inline std::size_t CacheManager::get_num_pages()   const noexcept { return m_num_pages; }
inline ov::element::Type CacheManager::get_element_type() const noexcept { return m_elem_type; }

template <typename T>
std::vector<std::size_t> CacheManager::insert(Handle h,
                                              const T* key_src,
                                              const T* value_src,
                                              std::size_t page_count,
                                              const T* scores) {
    if (!is_element_compatible_with_T(m_elem_type, sizeof(T))) {
        throw std::runtime_error("CacheManager::insert<T>: T does not match element type");
    }
    if (page_count == 0) return {};

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

    debug_trace("[CM] insert: h=%llu pages=%zu\n",
                static_cast<unsigned long long>(h), page_idxs.size());

    return page_idxs;
}

template <typename T>
void CacheManager::set_page_scores(Handle h,
                                   const std::vector<std::size_t>& page_indices,
                                   const T* scores) {
    if (!is_element_compatible_with_T(m_elem_type, sizeof(T))) {
        throw std::runtime_error("CacheManager::set_page_scores<T>: T does not match element type");
    }
    if (page_indices.empty()) return;

    std::lock_guard<std::mutex> lk(m_mu);

    std::vector<float> fs(page_indices.size());
    for (std::size_t i = 0; i < page_indices.size(); ++i) {
        fs[i] = cast_score_to_float(m_elem_type, static_cast<const void*>(&scores[i]));
    }
    set_scores_for_pages_unlocked(h, page_indices, fs.data());

    debug_trace("[CM] set_page_scores: h=%llu count=%zu\n",
                static_cast<unsigned long long>(h), page_indices.size());
}

} // namespace internal
} // namespace ov
