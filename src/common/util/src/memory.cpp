// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/memory.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

#include "openvino/util/common_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::util {

namespace {

/**
 * @brief Functor that touches one byte per page over [m_begin, m_end) to force the pages resident.
 *
 * The volatile accumulator prevents the compiler from optimizing the read loop away.
 */
struct PageToucher {
    const uint8_t* m_begin;
    const uint8_t* m_end;
    const size_t m_page_size;

    void operator()() const noexcept {
        volatile uint8_t local = 0;  // prevents the compiler from optimizing the loop away
        for (auto begin = m_begin; begin < m_end; begin += m_page_size) {
            local += *begin;
        }
    }
};

void populate_pages(void* ptr, size_t size, size_t num_threads) noexcept {
    // ptr and size are guaranteed page-aligned by vm_prefetch's precondition.
    const auto page_size = static_cast<size_t>(get_system_page_size());
    const auto chunk_size = std::max<size_t>(align_size_up(size / num_threads, page_size), one_mib);

    std::vector<std::thread> threads;
    threads.reserve(ceil_div(size, chunk_size));

    for (auto first = reinterpret_cast<const uint8_t*>(ptr), last = first + size; first < last; first += chunk_size) {
        threads.emplace_back(PageToucher{first, std::min(first + chunk_size, last), page_size});
    }
    for (auto& t : threads) {
        t.join();
    }
}

}  // namespace

AlignedRegion make_hint_region(const void* data, size_t mapping_size, size_t offset, size_t size) noexcept {
    const auto page_size = static_cast<size_t>(get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};
    }
    const auto available = mapping_size - offset;
    const auto raw_len = (size == auto_size) ? available : std::min(size, available);
    return align_region(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
}

void prefetch_mapped_region(const void* data, size_t mapping_size, size_t offset, size_t size) noexcept {
    // Below 4 MiB the overhead of spawning threads exceeds the benefit; skip.
    if (const auto region = make_hint_region(data, mapping_size, offset, size); region.m_length > 4 * one_mib) {
        const auto num_threads = std::min<size_t>(10, std::thread::hardware_concurrency());
        const auto aligned_size = align_size_up(region.m_length, static_cast<size_t>(get_system_page_size()));
        vm_prefetch(reinterpret_cast<void*>(region.m_address), aligned_size, num_threads);
    }
}

void vm_prefetch(void* ptr, size_t size, size_t num_threads) noexcept {
    assert(ptr != nullptr && size > 0);
    if (num_threads == 0) {
        vm_prefetch_hint(ptr, size);
    } else {
        // blocks until every page has been faulted in.
        populate_pages(ptr, size, num_threads);
    }
}

}  // namespace ov::util
