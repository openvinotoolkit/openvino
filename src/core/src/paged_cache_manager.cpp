

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#    include <malloc.h>
#endif

#include "openvino/core/paged_cache_manager.hpp"

namespace ov {
namespace internal {

// aligned allocation
static void* aligned_alloc_portable(std::size_t alignment, std::size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
#endif
}

static void aligned_free_portable(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

PagedCacheManager::PagedCacheManager(ov::element::Type elem_type,
                                     std::size_t total_bytes,
                                     std::size_t page_bytes,
                                     std::size_t alignment_bytes)
    : m_elem_type(elem_type),
      m_total_bytes(total_bytes),
      m_page_bytes(page_bytes),
      m_alignment_bytes(alignment_bytes),
      m_num_pages((page_bytes == 0) ? 0 : (total_bytes / page_bytes)) {
    if (page_bytes == 0 || total_bytes == 0 || (total_bytes % page_bytes) != 0) {
        throw std::runtime_error("PagedCacheManager: total_bytes must be a positive multiple of page_bytes");
    }
    if (alignment_bytes == 0 || (alignment_bytes & (alignment_bytes - 1)) != 0) {
        throw std::runtime_error("PagedCacheManager: alignment_bytes must be a power of two");
    }

    m_key_base = aligned_alloc_portable(m_alignment_bytes, m_total_bytes);
    m_value_base = aligned_alloc_portable(m_alignment_bytes, m_total_bytes);
    if (!m_key_base || !m_value_base) {
        aligned_free_portable(m_key_base);
        aligned_free_portable(m_value_base);
        throw std::runtime_error("PagedCacheManager: aligned allocation failed");
    }

    m_pages.resize(m_num_pages);
    for (std::size_t i = 0; i < m_num_pages; ++i) {
        m_pages[i].index = i;
        m_pages[i].score = std::numeric_limits<float>::infinity();
        m_pages[i].owner = 0;
        m_free_page_list.push_back(i);
    }
    rebuild_evict_heap_unlocked();

    debug_trace("[CM] ctor: elem=%s total=%zu page=%zu align=%zu pages=%zu\n",
                m_elem_type.get_type_name().c_str(),
                m_total_bytes,
                m_page_bytes,
                m_alignment_bytes,
                m_num_pages);
}

PagedCacheManager::~PagedCacheManager() {
    debug_trace("[CM] dtor\n");
    aligned_free_portable(m_key_base);
    aligned_free_portable(m_value_base);
}

// registration
PagedCacheManager::handle_t PagedCacheManager::register_operator(const ov::Output<ov::Node>& pa_output) {
    std::lock_guard<std::mutex> lk(m_mu);
    const handle_t h = m_next_handle.fetch_add(1, std::memory_order_relaxed);

    operator_state st;
    st.node_output = pa_output;
    compute_subsequence_begins_unlocked(st);
    m_ops.emplace(h, std::move(st));

    debug_trace("[CM] register op: h=%llu\n", static_cast<unsigned long long>(h));
    return h;
}

void PagedCacheManager::unregister_operator(handle_t h) {
    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_ops.find(h);
    if (it == m_ops.end())
        return;

    for (std::size_t pidx : it->second.pages) {
        if (pidx < m_pages.size() && m_pages[pidx].owner == h) {
            m_pages[pidx].owner = 0;
            m_pages[pidx].score = std::numeric_limits<float>::infinity();
            m_free_page_list.push_back(pidx);
        }
    }
    m_ops.erase(it);
    rebuild_evict_heap_unlocked();
    debug_trace("[CM] unregister op: h=%llu\n", static_cast<unsigned long long>(h));
}

// buffers
PagedCacheManager::cache_blocks PagedCacheManager::get_cache_blocks() const noexcept {
    std::lock_guard<std::mutex> lk(m_mu);
    return cache_blocks{m_key_base, m_value_base, m_total_bytes, m_total_bytes};
}

// per-operator metadata
PagedCacheManager::subsequence_view PagedCacheManager::get_subsequence_begins(handle_t h) const {
    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_ops.find(h);
    if (it == m_ops.end())
        return {};
    const auto& v = it->second.subseq_begins;
    return subsequence_view{v.data(), v.size()};
}

void PagedCacheManager::set_operator_cache_geometry(handle_t h,
                                                    std::size_t num_blocks,
                                                    std::size_t num_heads,
                                                    std::size_t block_size,
                                                    std::size_t key_head_size,
                                                    std::size_t value_head_size,
                                                    std::size_t query_head_size) {
    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_ops.find(h);
    if (it == m_ops.end())
        return;
    auto& st = it->second;
    st.num_blocks = num_blocks;
    st.num_heads = num_heads;
    st.block_size = block_size;
    st.key_head_size = key_head_size;
    st.value_head_size = value_head_size;
    st.query_head_size = query_head_size;
    debug_trace("[CM] set_operator_cache_geometry: h=%llu blocks=%zu heads=%zu blk=%zu\n",
                static_cast<unsigned long long>(h),
                num_blocks,
                num_heads,
                block_size);
}

// page span
PagedCacheManager::page_span PagedCacheManager::get_page_span(handle_t h,
                                                              std::size_t start_page,
                                                              std::size_t page_count) const {
    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_ops.find(h);
    if (it == m_ops.end() || page_count == 0)
        return {};

    const auto& owned = it->second.pages;
    if (owned.empty())
        return {};

    std::vector<std::size_t> sorted = owned;
    std::sort(sorted.begin(), sorted.end());

    auto f = std::lower_bound(sorted.begin(), sorted.end(), start_page);
    if (f == sorted.end())
        return {};

    std::size_t first = *f;
    std::size_t want_end = start_page + page_count;
    std::size_t last = first;

    auto itrun = f;
    ++itrun;
    while (itrun != sorted.end() && *itrun == last + 1 && *itrun < want_end) {
        last = *itrun;
        ++itrun;
    }

    const std::size_t pages_in_span = (last - first + 1);
    const std::size_t bytes = pages_in_span * m_page_bytes;

    return page_span{first * m_page_bytes, first * m_page_bytes, bytes};
}

// page mgmt
std::vector<std::size_t> PagedCacheManager::acquire_pages(handle_t h, std::size_t page_count) {
    std::lock_guard<std::mutex> lk(m_mu);
    return acquire_pages_unlocked(h, page_count);
}

void PagedCacheManager::release_pages(handle_t h, const std::vector<std::size_t>& pages) {
    std::lock_guard<std::mutex> lk(m_mu);
    auto it = m_ops.find(h);
    if (it == m_ops.end())
        return;
    auto& st = it->second;

    for (std::size_t pidx : pages) {
        if (pidx >= m_pages.size())
            continue;
        if (m_pages[pidx].owner != h)
            continue;

        auto pit = std::find(st.pages.begin(), st.pages.end(), pidx);
        if (pit != st.pages.end()) {
            const auto pos = static_cast<std::size_t>(std::distance(st.pages.begin(), pit));
            st.pages.erase(pit);
            if (pos < st.scores.size())
                st.scores.erase(st.scores.begin() + static_cast<std::ptrdiff_t>(pos));
        }

        m_pages[pidx].owner = 0;
        m_pages[pidx].score = std::numeric_limits<float>::infinity();
        m_free_page_list.push_back(pidx);
    }
    rebuild_evict_heap_unlocked();
    debug_trace("[CM] release_pages: h=%llu count=%zu\n", static_cast<unsigned long long>(h), pages.size());
}

// insert & scoring helpers
std::vector<std::size_t> PagedCacheManager::acquire_pages_unlocked(handle_t h, std::size_t page_count) {
    if (page_count == 0)
        return {};
    auto it = m_ops.find(h);
    if (it == m_ops.end())
        throw std::runtime_error("PagedCacheManager::acquire_pages_unlocked: unknown handle");
    auto& st = it->second;

    ensure_free_pages_unlocked(page_count);

    std::vector<std::size_t> granted;
    granted.reserve(page_count);
    for (std::size_t i = 0; i < page_count; ++i) {
        if (m_free_page_list.empty()) {
            evict_one_unlocked();
        }
        if (m_free_page_list.empty())
            break;

        const std::size_t pidx = m_free_page_list.front();
        m_free_page_list.pop_front();

        m_pages[pidx].owner = h;
        m_pages[pidx].score = std::numeric_limits<float>::infinity();

        st.pages.push_back(pidx);
        st.scores.push_back(std::numeric_limits<float>::infinity());
        granted.push_back(pidx);
    }

    rebuild_evict_heap_unlocked();
    debug_trace("[CM] acquire_pages: h=%llu count=%zu free_after=%zu\n",
                static_cast<unsigned long long>(h),
                page_count,
                m_free_page_list.size());
    return granted;
}

void PagedCacheManager::copy_pages_into_buffers_unlocked(const void* key_src_bytes,
                                                         const void* value_src_bytes,
                                                         const std::vector<std::size_t>& page_idxs) {
    if (page_idxs.empty())
        return;
    const auto* ksrc = static_cast<const unsigned char*>(key_src_bytes);
    const auto* vsrc = static_cast<const unsigned char*>(value_src_bytes);
    const std::size_t bytes_per_page = m_page_bytes;

    for (std::size_t i = 0; i < page_idxs.size(); ++i) {
        const std::size_t p = page_idxs[i];
        std::memcpy(static_cast<unsigned char*>(offset_key(p)), ksrc + i * bytes_per_page, bytes_per_page);
        std::memcpy(static_cast<unsigned char*>(offset_value(p)), vsrc + i * bytes_per_page, bytes_per_page);
    }
}

void PagedCacheManager::set_scores_for_pages_unlocked(handle_t h,
                                                      const std::vector<std::size_t>& page_idxs,
                                                      const float* scores) {
    auto it = m_ops.find(h);
    if (it == m_ops.end())
        throw std::runtime_error("PagedCacheManager::set_scores_for_pages_unlocked: unknown handle");
    auto& st = it->second;

    for (std::size_t i = 0; i < page_idxs.size(); ++i) {
        const std::size_t pidx = page_idxs[i];
        if (pidx >= m_pages.size())
            continue;
        if (m_pages[pidx].owner != h)
            continue;

        auto pit = std::find(st.pages.begin(), st.pages.end(), pidx);
        if (pit != st.pages.end()) {
            const auto pos = static_cast<std::size_t>(std::distance(st.pages.begin(), pit));
            if (pos < st.scores.size())
                st.scores[pos] = scores[i];
        }
        m_pages[pidx].score = scores[i];
    }
    rebuild_evict_heap_unlocked();
}

// eviction
void PagedCacheManager::ensure_free_pages_unlocked(std::size_t need_pages) {
    if (m_free_page_list.size() >= need_pages)
        return;
    const std::size_t deficit = need_pages - m_free_page_list.size();
    for (std::size_t i = 0; i < deficit; ++i) {
        evict_one_unlocked();
        if (m_free_page_list.size() >= need_pages)
            break;
    }
}

void PagedCacheManager::evict_one_unlocked() {
    if (m_evict_heap.empty())
        rebuild_evict_heap_unlocked();
    if (m_evict_heap.empty())
        return;

    std::pop_heap(m_evict_heap.begin(), m_evict_heap.end(), [this](std::size_t a, std::size_t b) {
        return heap_less_(m_pages, a, b);
    });
    const std::size_t victim = m_evict_heap.back();
    m_evict_heap.pop_back();

    if (victim >= m_pages.size())
        return;
    auto& pg = m_pages[victim];
    if (pg.owner == 0)
        return;

    const handle_t owner = pg.owner;
    auto it = m_ops.find(owner);
    if (it != m_ops.end()) {
        auto& st = it->second;
        auto pit = std::find(st.pages.begin(), st.pages.end(), victim);
        if (pit != st.pages.end()) {
            const auto pos = static_cast<std::size_t>(std::distance(st.pages.begin(), pit));
            st.pages.erase(pit);
            if (pos < st.scores.size())
                st.scores.erase(st.scores.begin() + static_cast<std::ptrdiff_t>(pos));
        }
    }

    pg.owner = 0;
    pg.score = std::numeric_limits<float>::infinity();
    m_free_page_list.push_back(victim);

    debug_trace("[CM] evict_one: page=%zu free_now=%zu\n", victim, m_free_page_list.size());
}

void PagedCacheManager::evict_to_target_free(std::size_t target_free_pages) {
    std::lock_guard<std::mutex> lk(m_mu);
    while (m_free_page_list.size() < target_free_pages) {
        evict_one_unlocked();
        if (m_evict_heap.empty())
            break;
    }
}

// snapshot
PagedCacheManager::snapshot_t PagedCacheManager::snapshot() const {
    std::lock_guard<std::mutex> lk(m_mu);
    snapshot_t s;
    s.elem_type = m_elem_type;
    s.total_bytes = m_total_bytes;
    s.page_bytes = m_page_bytes;
    s.alignment_bytes = m_alignment_bytes;
    s.num_pages = m_num_pages;
    s.free_pages = m_free_page_list.size();

    s.ops.reserve(m_ops.size());
    for (const auto& kv : m_ops) {
        snapshot_t::op_info oi;
        oi.handle = kv.first;
        oi.pages = kv.second.pages;
        oi.scores = kv.second.scores;
        oi.subseq_begins = kv.second.subseq_begins;
        oi.num_blocks = kv.second.num_blocks;
        oi.num_heads = kv.second.num_heads;
        oi.block_size = kv.second.block_size;
        oi.key_head_size = kv.second.key_head_size;
        oi.value_head_size = kv.second.value_head_size;
        oi.query_head_size = kv.second.query_head_size;
        s.ops.push_back(std::move(oi));
    }
    return s;
}

// debug
void PagedCacheManager::debug_trace(const char* fmt, ...) noexcept {
#if CM_DEBUG
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
#else
    (void)fmt;
#endif
}

// priv helpers
void* PagedCacheManager::offset_key(std::size_t page_idx) const noexcept {
    return static_cast<void*>(static_cast<unsigned char*>(m_key_base) + page_idx * m_page_bytes);
}

void* PagedCacheManager::offset_value(std::size_t page_idx) const noexcept {
    return static_cast<void*>(static_cast<unsigned char*>(m_value_base) + page_idx * m_page_bytes);
}

void PagedCacheManager::compute_subsequence_begins_unlocked(operator_state& st) const {
    (void)st;  // hook for real PA wiring if needed
}

float PagedCacheManager::cast_score_to_float(ov::element::Type et, const void* src_scalar) noexcept {
    switch (et) {
    case ov::element::f32:
        return *static_cast<const float*>(src_scalar);
    case ov::element::f16: {
        const uint16_t u = *static_cast<const uint16_t*>(src_scalar);
        const uint32_t s = (u >> 15) & 1u;
        const uint32_t e = (u >> 10) & 0x1fu;
        const uint32_t f = u & 0x3ffu;
        float out;
        if (e == 0)
            out = std::ldexp(static_cast<float>(f), -24);
        else if (e != 31)
            out = std::ldexp(static_cast<float>(f + 1024), static_cast<int>(e) - 25);
        else
            out = f ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
        return s ? -out : out;
    }
    case ov::element::bf16: {
        const uint16_t u = *static_cast<const uint16_t*>(src_scalar);
        const uint32_t v = static_cast<uint32_t>(u) << 16;
        float f;
        std::memcpy(&f, &v, sizeof(float));
        return f;
    }
    case ov::element::i32:
        return static_cast<float>(*static_cast<const int32_t*>(src_scalar));
    case ov::element::i64:
        return static_cast<float>(*static_cast<const int64_t*>(src_scalar));
    case ov::element::u32:
        return static_cast<float>(*static_cast<const uint32_t*>(src_scalar));
    case ov::element::u64:
        return static_cast<float>(*static_cast<const uint64_t*>(src_scalar));
    default:
        return *static_cast<const float*>(src_scalar);
    }
}

bool PagedCacheManager::is_element_compatible_with_T(ov::element::Type et, size_t sizeofT) noexcept {
    switch (et) {
    case ov::element::f32:
        return sizeofT == 4;
    case ov::element::bf16:
        return sizeofT == 2;
    case ov::element::f16:
        return sizeofT == 2;
    case ov::element::i32:
        return sizeofT == 4;
    case ov::element::i64:
        return sizeofT == 8;
    case ov::element::u32:
        return sizeofT == 4;
    case ov::element::u64:
        return sizeofT == 8;
    default:
        return sizeofT == 4;
    }
}

void PagedCacheManager::rebuild_evict_heap_unlocked() {
    m_evict_heap.clear();
    m_evict_heap.reserve(m_pages.size());
    for (const auto& pg : m_pages) {
        if (pg.owner != 0 && std::isfinite(pg.score)) {
            m_evict_heap.push_back(pg.index);
        }
    }
    std::make_heap(m_evict_heap.begin(), m_evict_heap.end(), [this](std::size_t a, std::size_t b) {
        return heap_less_(m_pages, a, b);
    });
}

bool PagedCacheManager::heap_less_(const std::vector<page_t>& pages, std::size_t a, std::size_t b) noexcept {
    const float sa = (a < pages.size()) ? pages[a].score : std::numeric_limits<float>::infinity();
    const float sb = (b < pages.size()) ? pages[b].score : std::numeric_limits<float>::infinity();
    return sa > sb;  // lower score = higher eviction priority
}

}  // namespace internal
}  // namespace ov
