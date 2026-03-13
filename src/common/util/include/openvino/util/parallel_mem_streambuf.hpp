// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <streambuf>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

#include "openvino/core/parallel.hpp"

namespace ov {
namespace util {

/// @brief A std::streambuf that reads from an in-memory buffer using parallel
///        memcpy for large reads.
///
/// Intended for mmap-backed tensors: the tensor's raw memory is already mapped
/// into the process but pages may not yet be resident.  For large reads,
/// splitting the copy across N threads triggers concurrent page faults, raising
/// the OS I/O queue depth and saturating NVMe bandwidth.
///
/// On Windows, after each large copy the consumed source pages are released
/// from the process working-set via VirtualFree(MEM_RESET) to relieve RAM
/// pressure when loading multi-GB models.
///
/// Usage:
/// @code
///   // In plugin::import_model(const ov::Tensor& model, ...):
///   ov::util::ParallelMemStreamBuf par_buf(model.data(), model.get_byte_size());
///   std::istream stream(&par_buf);
///   // pass stream to BinaryInputBuffer or any std::istream& consumer
/// @endcode
class ParallelMemStreamBuf : public std::streambuf {
public:
    static constexpr size_t DEFAULT_THRESHOLD = 4UL * 1024 * 1024;  // 4 MB

    /// @param data       Pointer to the start of the memory region.
    /// @param size       Total size of the memory region in bytes.
    /// @param threshold  Minimum read size to engage parallel memcpy.
    ParallelMemStreamBuf(const void* data, size_t size, size_t threshold = DEFAULT_THRESHOLD)
        : m_begin(static_cast<const char*>(data)),
          m_end(static_cast<const char*>(data) + size),
          m_current(static_cast<const char*>(data)),
          m_threshold(threshold) {}

    ~ParallelMemStreamBuf() override = default;

    ParallelMemStreamBuf(const ParallelMemStreamBuf&) = delete;
    ParallelMemStreamBuf& operator=(const ParallelMemStreamBuf&) = delete;

protected:
    // -----------------------------------------------------------------------
    // xsgetn: hot path — called by sgetn() for all bulk reads
    // -----------------------------------------------------------------------
    std::streamsize xsgetn(char_type* dst, std::streamsize n) override {
        if (m_current >= m_end) {
            return 0;
        }
        const std::streamsize avail = static_cast<std::streamsize>(m_end - m_current);
        const std::streamsize to_copy = std::min(n, avail);

        if (static_cast<size_t>(to_copy) >= m_threshold) {
            parallel_copy(dst, m_current, static_cast<size_t>(to_copy));
        } else {
            std::memcpy(dst, m_current, static_cast<size_t>(to_copy));
        }

        m_current += to_copy;
        return to_copy;
    }

    // -----------------------------------------------------------------------
    // underflow: single-char peek path
    // -----------------------------------------------------------------------
    int_type underflow() override {
        if (m_current >= m_end) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*m_current);
    }

    int_type uflow() override {
        if (m_current >= m_end) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*m_current++);
    }

    // -----------------------------------------------------------------------
    // Seek support
    // -----------------------------------------------------------------------
    pos_type seekoff(off_type off,
                     std::ios_base::seekdir way,
                     std::ios_base::openmode /* which */) override {
        const char* new_pos = nullptr;
        if (way == std::ios_base::beg) {
            new_pos = m_begin + off;
        } else if (way == std::ios_base::cur) {
            new_pos = m_current + off;
        } else {
            new_pos = m_end + off;
        }

        if (new_pos < m_begin || new_pos > m_end) {
            return pos_type(off_type(-1));
        }

        m_current = new_pos;
        return pos_type(static_cast<off_type>(m_current - m_begin));
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode /* which */) override {
        return seekoff(off_type(pos), std::ios_base::beg, std::ios_base::in);
    }

    std::streamsize showmanyc() override {
        const std::streamsize avail = static_cast<std::streamsize>(m_end - m_current);
        return avail > 0 ? avail : -1;
    }

private:
    void parallel_copy(char* dst, const char* src, size_t size) {
        constexpr size_t MIN_CHUNK = 2UL * 1024 * 1024;  // 2 MB minimum per thread
        const size_t num_chunks = std::max(size_t{1}, size / MIN_CHUNK);
        const size_t chunk_size = (size + num_chunks - 1) / num_chunks;

#ifdef _WIN32
        // Prefetch: trigger page faults up-front to maximise NVMe queue depth.
        WIN32_MEMORY_RANGE_ENTRY prefetch_range{const_cast<char*>(src), size};
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &prefetch_range, 0);
#endif

        ov::parallel_for(num_chunks, [&](size_t i) {
            const size_t offset = i * chunk_size;
            const size_t copy_size = (i + 1 == num_chunks) ? (size - offset) : chunk_size;
            std::memcpy(dst + offset, src + offset, copy_size);
        });

#ifdef _WIN32
        // Release consumed mmap pages from the working-set to avoid RAM pressure
        // when loading multi-GB models. MEM_RESET marks pages as no longer needed;
        // the kernel may reclaim them without writing to the page file.
        constexpr uintptr_t PAGE_MASK = ~static_cast<uintptr_t>(4095u);
        const char* reset_begin =
            reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(src) & PAGE_MASK);
        const char* reset_end = reinterpret_cast<const char*>(
            (reinterpret_cast<uintptr_t>(src) + size) & PAGE_MASK);
        if (reset_begin < reset_end) {
            VirtualFree(const_cast<char*>(reset_begin),
                        static_cast<SIZE_T>(reset_end - reset_begin),
                        MEM_RESET);
        }
#endif
    }

    const char* m_begin;
    const char* m_end;
    const char* m_current;
    size_t m_threshold;
};

}  // namespace util
}  // namespace ov
