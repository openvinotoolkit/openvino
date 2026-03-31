// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parallel_mem_streambuf.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu {

ParallelMemStreamBuf::ParallelMemStreamBuf(const void* data, size_t size, size_t threshold)
    : m_begin(static_cast<const char*>(data)),
      m_end(static_cast<const char*>(data) + size),
      m_current(static_cast<const char*>(data)),
      m_threshold(threshold) {
    // Detect whether this memory is a file-backed mmap region.
    // If so, build a ParallelReadStreamBuf over the same file+offset so
    // direct reads are used instead of mmap+memcpy.  This avoids 2x RAM
    // pressure (mmap working-set + destination buffer) that causes
    // working-set thrashing for multi-GB models.
    if (size >= threshold) {
        std::filesystem::path file_path;
        std::streamoff file_off = 0;
        if (ov::util::get_mmap_file_info(data, file_path, file_off)) {
            try {
                m_file_buf = std::make_unique<ov::util::ParallelReadStreamBuf>(file_path, file_off, threshold);
            } catch (...) {
                // File became inaccessible after mmap detection; fall through to memcpy path.
            }
        }
    }
    // For non-file-backed memory (anonymous mmap, USM host buffers, etc.)
    // fall back to async prefetch + parallel memcpy.
    if (!m_file_buf) {
        ov::util::prefetch_memory(data, size);
    }
}

std::streamsize ParallelMemStreamBuf::xsgetn(char_type* dst, std::streamsize n) {
    if (m_file_buf) {
        return m_file_buf->sgetn(dst, n);
    }
    if (n <= 0 || m_current >= m_end) {
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

ParallelMemStreamBuf::int_type ParallelMemStreamBuf::underflow() {
    if (m_file_buf) {
        return m_file_buf->sgetc();
    }
    if (m_current >= m_end) {
        return traits_type::eof();
    }
    return traits_type::to_int_type(*m_current);
}

ParallelMemStreamBuf::int_type ParallelMemStreamBuf::uflow() {
    if (m_file_buf) {
        return m_file_buf->sbumpc();
    }
    if (m_current >= m_end) {
        return traits_type::eof();
    }
    return traits_type::to_int_type(*m_current++);
}

ParallelMemStreamBuf::pos_type ParallelMemStreamBuf::seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) {
    if (m_file_buf) {
        return m_file_buf->pubseekoff(off, way, which);
    }
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

ParallelMemStreamBuf::pos_type ParallelMemStreamBuf::seekpos(pos_type pos, std::ios_base::openmode which) {
    if (m_file_buf) {
        return m_file_buf->pubseekpos(pos, which);
    }
    return seekoff(off_type(pos), std::ios_base::beg, std::ios_base::in);
}

std::streamsize ParallelMemStreamBuf::showmanyc() {
    if (m_file_buf) {
        return m_file_buf->in_avail();
    }
    const std::streamsize avail = static_cast<std::streamsize>(m_end - m_current);
    return avail > 0 ? avail : -1;
}

void ParallelMemStreamBuf::parallel_copy(char* dst, const char* src, size_t size) {
    // Cap threads: too many concurrent threads cause OS scheduling overhead
    // (Linux) or PFN-lock contention (Windows).  Use hardware_concurrency as
    // the upper bound, consistent with parallel_read.
    const size_t hw_conc = std::max(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t num_chunks = std::max(size_t{1}, std::min(size / ov::util::DEFAULT_PARALLEL_IO_MIN_CHUNK, hw_conc));
    const size_t chunk_size = (size + num_chunks - 1) / num_chunks;

    ov::util::prefetch_memory(src, size);

    std::vector<std::thread> workers;
    workers.reserve(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        try {
            workers.emplace_back([&, i]() {
                const size_t offset = i * chunk_size;
                const size_t copy_size = (i + 1 == num_chunks) ? (size - offset) : chunk_size;
                std::memcpy(dst + offset, src + offset, copy_size);
            });
        } catch (...) {
            for (auto& t : workers)
                t.join();
            const size_t done = i * chunk_size;
            if (done < size)
                std::memcpy(dst + done, src + done, size - done);
            return;
        }
    }
    for (auto& t : workers) {
        t.join();
    }
}

}  // namespace ov::intel_gpu
