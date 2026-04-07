// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_read_streambuf.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

ParallelReadStreamBuf::ParallelReadStreamBuf(const std::filesystem::path& path,
                                             std::streamoff header_offset,
                                             size_t threshold)
    : m_path(path),
      m_file_offset(header_offset),
      m_header_offset(header_offset),
      m_threshold(threshold) {
    get_file_handle_and_size(path, m_file_offset, m_handle, m_file_size);
}

ParallelReadStreamBuf::~ParallelReadStreamBuf() {
    close_file_handle(m_handle);
}

// xsgetn: main hot path - called by sgetn() for all bulk reads
std::streamsize ParallelReadStreamBuf::xsgetn(char_type* dst, std::streamsize n) {
    if (n <= 0)
        return 0;

    std::streamsize total = 0;

    // Drain any chars previously buffered by underflow()
    if (gptr() != nullptr && gptr() < egptr()) {
        const std::streamsize avail = static_cast<std::streamsize>(egptr() - gptr());
        const std::streamsize from_buf = std::min(n, avail);
        std::memcpy(dst, gptr(), static_cast<size_t>(from_buf));
        gbump(static_cast<int>(from_buf));
        total += from_buf;
        dst += from_buf;
        n -= from_buf;
    }

    if (n <= 0 || m_file_offset >= m_file_size) {
        return total;
    }

    const std::streamoff remaining = m_file_size - m_file_offset;
    const std::streamsize to_read = static_cast<std::streamsize>(std::min(static_cast<std::streamoff>(n), remaining));

    const size_t bytes = static_cast<size_t>(to_read);
    const size_t offset = static_cast<size_t>(m_file_offset);

    bool ok = (bytes >= m_threshold) ? parallel_read(dst, bytes, offset) : single_read(dst, bytes, offset);

    if (ok) {
        m_file_offset += to_read;
        total += to_read;
    }

    return total;
}

// underflow: called for single-char peek / non-bulk reads (e.g. std::getline)
ParallelReadStreamBuf::int_type ParallelReadStreamBuf::underflow() {
    if (m_file_offset >= m_file_size) {
        return traits_type::eof();
    }
    if (!m_underflow_buf) {
        m_underflow_buf = std::make_unique<char_type[]>(UNDERFLOW_BUF);
    }
    // Read a batch of up to UNDERFLOW_BUF bytes so that character-by-character
    // consumers (std::getline, operator>>) don't issue one pread per char.
    const size_t to_read =
        static_cast<size_t>(std::min(static_cast<std::streamoff>(UNDERFLOW_BUF), m_file_size - m_file_offset));
    if (!single_read(m_underflow_buf.get(), to_read, static_cast<size_t>(m_file_offset))) {
        return traits_type::eof();
    }
    // Advance m_file_offset past the bytes we just read into the get area.
    // m_file_offset now points to the byte after egptr(), consistent with
    // the seekoff(0, cur) formula: logical_pos = m_file_offset - (egptr - gptr).
    m_file_offset += static_cast<std::streamoff>(to_read);
    setg(m_underflow_buf.get(), m_underflow_buf.get(), m_underflow_buf.get() + to_read);
    return traits_type::to_int_type(m_underflow_buf[0]);
}

ParallelReadStreamBuf::pos_type ParallelReadStreamBuf::seekoff(off_type off,
                                                               std::ios_base::seekdir way,
                                                               std::ios_base::openmode /* which */) {
    // All internal positions (m_file_offset, m_file_size, m_header_offset) are
    // absolute byte offsets from the start of the file.  The public-facing
    // stream positions are *logical* offsets: 0 == header_offset in the file.
    std::streamoff new_pos = 0;
    if (way == std::ios_base::beg) {
        // off is a logical offset; translate to absolute file offset.
        new_pos = m_header_offset + off;
    } else if (way == std::ios_base::cur) {
        // Account for the buffered chars from underflow() not yet consumed.
        const std::streamsize ahead = (gptr() != nullptr) ? static_cast<std::streamsize>(egptr() - gptr()) : 0;
        new_pos = m_file_offset - ahead + off;  // stays absolute
        // Pure tell (off == 0): return current position without any side effects
        // on the get area or m_file_offset.  Discarding the underflow buffer on a
        // tell would force the next read to re-issue a pread for data that is
        // already buffered, breaking interleaved getline()+tellg() patterns.
        if (off == 0) {
            if (new_pos < m_header_offset || new_pos > m_file_size)
                return pos_type(off_type(-1));
            return pos_type(new_pos - m_header_offset);
        }
    } else {
        new_pos = m_file_size + off;  // stays absolute
    }

    // Reject seeks before the logical stream start or past the file end.
    if (new_pos < m_header_offset || new_pos > m_file_size) {
        return pos_type(off_type(-1));
    }

    setg(nullptr, nullptr, nullptr);  // invalidate get-area
    m_file_offset = new_pos;
    // Return the logical position (0 == start of exposed stream).
    return pos_type(m_file_offset - m_header_offset);
}

ParallelReadStreamBuf::pos_type ParallelReadStreamBuf::seekpos(pos_type pos, std::ios_base::openmode /* which */) {
    return seekoff(off_type(pos), std::ios_base::beg, std::ios_base::in);
}

std::streamsize ParallelReadStreamBuf::showmanyc() {
    // Report both buffered characters (in the get area) and remaining
    // bytes in the underlying file. Return -1 only when nothing more is
    // available, to match std::streambuf expectations.
    std::streamsize buffered = 0;
    if (gptr() != nullptr && egptr() != nullptr && egptr() > gptr()) {
        buffered = static_cast<std::streamsize>(egptr() - gptr());
    }
    std::streamoff remaining_off = m_file_size - m_file_offset;
    if (remaining_off < 0) {
        remaining_off = 0;
    }
    const std::streamsize remaining = remaining_off > 0 ? static_cast<std::streamsize>(remaining_off) : 0;
    const std::streamsize total = buffered + remaining;
    return total > 0 ? total : static_cast<std::streamsize>(-1);
}

// Single-threaded positional read
bool ParallelReadStreamBuf::single_read(char* dst, size_t size, size_t file_offset) {
    return positional_read(m_handle, dst, size, file_offset);
}

// Parallel positional read
bool ParallelReadStreamBuf::parallel_read(char* dst, size_t size, size_t file_offset) {
    const size_t hw_threads = std::max(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t max_by_size = size / (1024 * 1024);  // 1 thread per MB
    const size_t num_threads = std::max(size_t{1}, std::min(hw_threads, max_by_size));

    if (num_threads == 1) {
        return single_read(dst, size, file_offset);
    }

    // Round chunk_size UP to the next 4 KiB boundary so that every thread's
    // start offset is page-aligned (better I/O coalescing on NVMe/direct I/O).
    // Because rounding up means num_threads * chunk_size >= size, two extra
    // guards are required:
    //   1. Non-last threads: cap read to min(chunk_size, size - cur_offset) so
    //      they never stride past EOF when the aligned chunk extends beyond it.
    //   2. Last thread: use (size - cur_offset) to capture every remaining byte
    //      including the fragment that lies beyond (num_threads-1) * chunk_size
    //      but before size.  Using chunk_size here would silently drop those bytes.
    size_t chunk_size = size / num_threads;
    chunk_size = (chunk_size + 4095u) & ~size_t{4095u};

    std::atomic<bool> success{true};
    // Each worker opens its own file descriptor so that Linux's per-file-
    // description readahead state (file_ra_state / f_ra) is independent per
    // thread. Sharing a single fd causes concurrent pread() calls to corrupt
    // each other's sequential readahead prediction, collapsing throughput from
    // ~3.5 GB/s sequential to ~0.5 GB/s.
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (size_t ithr = 0; ithr < num_threads; ++ithr) {
        try {
            workers.emplace_back([&, ithr]() {
                const size_t cur_offset = ithr * chunk_size;
                if (cur_offset >= size) {
                    return;  // page-alignment rounding created a surplus worker slot
                }
                // Last thread: read everything remaining (includes the fragment that
                // falls between (num_threads-1)*chunk_size and size).
                // Non-last threads: cap to min(chunk_size, size - cur_offset) so we
                // never read past eof when alignment pushed the chunk boundary beyond it.
                const size_t read_size =
                    (ithr == num_threads - 1) ? (size - cur_offset) : std::min(chunk_size, size - cur_offset);
                char* const ptr = dst + cur_offset;
                const size_t thread_file_offset = file_offset + cur_offset;

                FileHandle t_handle = open_file_for_read(m_path);
                if (t_handle == INVALID_HANDLE_VALUE) {
                    success = false;
                    return;
                }

                if (!positional_read(t_handle, ptr, read_size, thread_file_offset)) {
                    success = false;
                }
                close_file_handle(t_handle);
            });  // workers.emplace_back
        } catch (...) {
            success = false;
            break;
        }
    }
    for (auto& t : workers) {
        t.join();
    }
    return success.load();
}

}  // namespace ov::util
