// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <algorithm>
#include <istream>
#include <mutex>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/demand_pager.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/parallel_read_streambuf.hpp"

namespace ov {
namespace {

util::DemandPager& demand_pager() {
    // Intentionally leaked: LazyBuffer instances may still be destroyed during static destruction.
    static auto* const instance = new util::DemandPager();
    return *instance;
}

void on_page_fault(void* user_data, void* addr, size_t) noexcept {
    try {
        static_cast<const LazyBuffer*>(user_data)->hint_prefetch();
    } catch (...) {
        // An unresolved fault would block the faulting thread forever, so drop the registration
        // and let the pending access fail instead.
        demand_pager().unregister_region(addr);
    }
}
}  // namespace

LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_loaded{false} {
    m_byte_size = byte_size;

    const auto file_size = util::file_size(m_file_path);
    OPENVINO_ASSERT(file_size >= 0, "Failed to get file size for ", m_file_path);

    const bool offset_fits = m_offset <= static_cast<size_t>(file_size);
    const bool size_fits = m_byte_size <= static_cast<size_t>(file_size) - m_offset;
    OPENVINO_ASSERT(offset_fits && size_fits,
                    "Requested region ",
                    m_offset,
                    "+",
                    m_byte_size,
                    " exceeds file size ",
                    file_size,
                    " for file: ",
                    m_file_path);

    // Page faults are delivered for whole pages, so the reservation is rounded up while size() stays as requested.
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    m_mapped_size = util::align_size_up(std::max<size_t>(1, m_byte_size), page_size);

    m_aligned_buffer = static_cast<char*>(demand_pager().reserve(m_mapped_size));
    OPENVINO_ASSERT(m_aligned_buffer != nullptr, "Failed to reserve memory for LazyBuffer: ", m_file_path);

    m_delegated = demand_pager().register_region(&on_page_fault, this, m_aligned_buffer, m_mapped_size);
    if (!m_delegated) {
        // Without fault delegation the reservation reads as zeros, so the data has to be loaded up front.
        try {
            hint_prefetch();
        } catch (...) {
            demand_pager().release(m_aligned_buffer, m_mapped_size);
            // The base destructor must not free memory which it did not allocate.
            m_aligned_buffer = nullptr;
            throw;
        }
    }
}

LazyBuffer::~LazyBuffer() {
    if (m_aligned_buffer) {
        demand_pager().unregister_region(m_aligned_buffer);
        demand_pager().release(m_aligned_buffer, m_mapped_size);
        m_aligned_buffer = nullptr;
    }
    m_byte_size = 0;
    m_mapped_size = 0;
}

LazyBuffer::LazyBuffer(LazyBuffer&& other) noexcept
    : AlignedBuffer(std::move(other)),
      m_file_path{std::move(other.m_file_path)},
      m_offset{std::exchange(other.m_offset, 0)},
      m_mapped_size{std::exchange(other.m_mapped_size, 0)},
      m_delegated{std::exchange(other.m_delegated, false)},
      m_loaded{other.m_loaded.exchange(false, std::memory_order_relaxed)} {
    demand_pager().update_user_data(m_aligned_buffer, this);
}

LazyBuffer& LazyBuffer::operator=(LazyBuffer&& other) noexcept {
    if (this != &other) {
        if (m_aligned_buffer) {
            demand_pager().unregister_region(m_aligned_buffer);
            demand_pager().release(m_aligned_buffer, m_mapped_size);
            m_aligned_buffer = nullptr;
        }
        AlignedBuffer::operator=(std::move(other));
        m_file_path = std::move(other.m_file_path);
        m_offset = std::exchange(other.m_offset, 0);
        m_mapped_size = std::exchange(other.m_mapped_size, 0);
        m_delegated = std::exchange(other.m_delegated, false);
        m_loaded = other.m_loaded.exchange(false, std::memory_order_relaxed);
        demand_pager().update_user_data(m_aligned_buffer, this);
    }
    return *this;
}

void LazyBuffer::hint_prefetch() const {
    if (m_loaded.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard lock{m_loading};
    if (m_loaded.load(std::memory_order_relaxed)) {
        return;
    }

    if (m_delegated) {
        // The page-rounded tail has to be zeroed because the whole reservation is installed at once.
        std::vector<char> staging(m_mapped_size, 0);
        read_file_data(staging.data());

        OPENVINO_ASSERT(demand_pager().populate(m_aligned_buffer, m_mapped_size, staging.data()),
                        "Failed to populate LazyBuffer: ",
                        m_file_path);
    } else {
        read_file_data(m_aligned_buffer);
    }
    m_loaded.store(true, std::memory_order_release);
}

void LazyBuffer::read_file_data(char* destination) const {
    util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
    std::istream file(&par_buf);
    OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
    file.read(destination, m_byte_size);
    OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
}

void LazyBuffer::hint_evict() noexcept {
    hint_evict(0, m_byte_size);
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (!m_delegated) {
        // Without fault delegation the pages cannot be repopulated on access, so dropping them would lose data.
        return;
    }

    if (m_loaded.load(std::memory_order_acquire)) {
        try {
            std::lock_guard lock{m_loading};
            if (m_loaded.load(std::memory_order_relaxed)) {
                demand_pager().evict(m_aligned_buffer, m_mapped_size);
                m_loaded.store(false, std::memory_order_release);
            }
        } catch (...) {
        }
    }
}
}  // namespace ov
