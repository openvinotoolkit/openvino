// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <streambuf>
#include <thread>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    ifdef min
#        undef min
#    endif
#    ifdef max
#        undef max
#    endif
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#define ENABLE_BD_PROFILING_LOG 1

namespace ov {
namespace util {

/// @brief A std::streambuf that reads from a file using parallel I/O for large
///        reads, bypassing the OS page cache pressure that mmap+memcpy incurs.
///
/// For reads >= threshold bytes, the read is split across N threads where each
/// thread issues its own positional read (pread on Linux, OVERLAPPED ReadFile on
/// Windows). Smaller reads fall through to a single positional call.
///
/// Usage:
/// @code
///   ParallelReadStreamBuf buf(cache_path, blob_offset_in_file);
///   std::istream stream(&buf);
///   cldnn::BinaryInputBuffer ib(stream, engine);
///   ib >> ...;
/// @endcode
class ParallelReadStreamBuf : public std::streambuf {
public:
    static constexpr size_t DEFAULT_THRESHOLD = 4UL * 1024 * 1024;  // 4 MB

    /// @param path           Path to the file to read.
    /// @param header_offset  Initial file position (seek-from-start value set at
    ///                       construction; does not affect seekg semantics).
    /// @param threshold      Minimum read size to trigger parallel I/O.
    explicit ParallelReadStreamBuf(const std::filesystem::path& path,
                                   std::streamoff header_offset = 0,
                                   size_t threshold = DEFAULT_THRESHOLD)
        : m_path(path),
          m_header_offset(header_offset),
          m_file_offset(header_offset),
          m_threshold(threshold) {
#ifdef _WIN32
        m_handle = CreateFileW(path.native().c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
        if (m_handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("ParallelReadStreamBuf: cannot open file: " + path.string());
        }
        LARGE_INTEGER file_size = {};
        if (!GetFileSizeEx(m_handle, &file_size)) {
            CloseHandle(m_handle);
            throw std::runtime_error("ParallelReadStreamBuf: cannot get file size: " + path.string());
        }
        m_file_size = static_cast<std::streamoff>(file_size.QuadPart);
#else
        m_fd = ::open(path.c_str(), O_RDONLY);
        if (m_fd == -1) {
            throw std::runtime_error("ParallelReadStreamBuf: cannot open file: " + path.string());
        }
        struct stat st = {};
        if (::fstat(m_fd, &st) != 0) {
            ::close(m_fd);
            throw std::runtime_error("ParallelReadStreamBuf: cannot stat file: " + path.string());
        }
        m_file_size = static_cast<std::streamoff>(st.st_size);
#endif
    }

    ~ParallelReadStreamBuf() override {
#ifdef _WIN32
        if (m_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(m_handle);
        }
#else
        if (m_fd != -1) {
            ::close(m_fd);
        }
#endif
    }

    ParallelReadStreamBuf(const ParallelReadStreamBuf&) = delete;
    ParallelReadStreamBuf& operator=(const ParallelReadStreamBuf&) = delete;

protected:
    // -----------------------------------------------------------------------
    // xsgetn: main hot path - called by sgetn() for all bulk reads
    // -----------------------------------------------------------------------
    std::streamsize xsgetn(char_type* dst, std::streamsize n) override {
        std::streamsize total = 0;

        // Drain any chars previously buffered by underflow()
        if (gptr() < egptr()) {
            const std::streamsize avail = static_cast<std::streamsize>(egptr() - gptr());
            const std::streamsize from_buf = std::min(n, avail);
            std::memcpy(dst, gptr(), static_cast<size_t>(from_buf));
            gbump(static_cast<int>(from_buf));
            // NOTE: m_file_offset was already advanced past egptr() in underflow().
            // Do NOT advance it again here.
            total += from_buf;
            dst += from_buf;
            n -= from_buf;
        }

        if (n <= 0 || m_file_offset >= m_file_size) {
            return total;
        }

        const std::streamoff remaining = m_file_size - m_file_offset;
        const std::streamsize to_read =
            static_cast<std::streamsize>(std::min(static_cast<std::streamoff>(n), remaining));

        const size_t bytes = static_cast<size_t>(to_read);
        const size_t offset = static_cast<size_t>(m_file_offset);

        bool ok = (bytes >= m_threshold) ? parallel_read(dst, bytes, offset) : single_read(dst, bytes, offset);

        if (ok) {
            m_file_offset += to_read;
            total += to_read;
        }

        return total;
    }

    // -----------------------------------------------------------------------
    // underflow: called for single-char peek / non-bulk reads (e.g. std::getline)
    // -----------------------------------------------------------------------
    int_type underflow() override {
        if (m_file_offset >= m_file_size) {
            return traits_type::eof();
        }
        // Read a batch of up to UNDERFLOW_BUF bytes so that character-by-character
        // consumers (std::getline, operator>>) don't issue one pread per char.
        const size_t to_read = static_cast<size_t>(
            std::min(static_cast<std::streamoff>(UNDERFLOW_BUF),
                     m_file_size - m_file_offset));
        if (!single_read(m_underflow_buf.data(), to_read, static_cast<size_t>(m_file_offset))) {
            return traits_type::eof();
        }
        // Advance m_file_offset past the bytes we just read into the get area.
        // m_file_offset now points to the byte after egptr(), consistent with
        // the seekoff(0, cur) formula: logical_pos = m_file_offset - (egptr - gptr).
        m_file_offset += static_cast<std::streamoff>(to_read);
        setg(m_underflow_buf.data(), m_underflow_buf.data(), m_underflow_buf.data() + to_read);
        return traits_type::to_int_type(m_underflow_buf[0]);
    }

    // -----------------------------------------------------------------------
    // Seek support
    // -----------------------------------------------------------------------
    pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode /* which */) override {
        std::streamoff new_pos = 0;
        if (way == std::ios_base::beg) {
            new_pos = off;
        } else if (way == std::ios_base::cur) {
            // Account for the buffered char from underflow() if it hasn't been consumed
            const std::streamsize ahead = static_cast<std::streamsize>(egptr() - gptr());
            new_pos = m_file_offset - ahead + off;
        } else {
            new_pos = m_file_size + off;
        }

        if (new_pos < 0 || new_pos > m_file_size) {
            return pos_type(off_type(-1));
        }

        setg(nullptr, nullptr, nullptr);  // invalidate get-area
        m_file_offset = new_pos;
        return pos_type(m_file_offset);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode /* which */) override {
        return seekoff(off_type(pos), std::ios_base::beg, std::ios_base::in);
    }

    std::streamsize showmanyc() override {
        const std::streamoff avail = m_file_size - m_file_offset;
        return avail > 0 ? static_cast<std::streamsize>(avail) : -1;
    }

private:
    // -----------------------------------------------------------------------
    // Single-threaded positional read
    // -----------------------------------------------------------------------
    bool single_read(char* dst, size_t size, size_t file_offset) {
#ifdef _WIN32
        char* cur = dst;
        size_t remaining = size;
        size_t cur_offset = file_offset;
        while (remaining > 0) {
            const DWORD to_read = static_cast<DWORD>(std::min(remaining, static_cast<size_t>(UINT_MAX - 1024u)));
            OVERLAPPED ov = {};
            ov.Offset = static_cast<DWORD>(cur_offset & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>((cur_offset >> 32) & 0xFFFFFFFFu);
            DWORD bytes_read = 0;
            if (!ReadFile(m_handle, cur, to_read, &bytes_read, &ov)) {
                if (GetLastError() != ERROR_IO_PENDING) {
                    return false;
                }
            }
            if (bytes_read == 0) {
                return false;
            }
            cur += bytes_read;
            cur_offset += bytes_read;
            remaining -= bytes_read;
        }
        return true;
#else
        char* cur = dst;
        size_t remaining = size;
        off_t cur_offset = static_cast<off_t>(file_offset);
        while (remaining > 0) {
            const ssize_t n = ::pread(m_fd, cur, remaining, cur_offset);
            if (n <= 0) {
                return false;
            }
            cur += n;
            cur_offset += n;
            remaining -= static_cast<size_t>(n);
        }
        return true;
#endif
    }

    // -----------------------------------------------------------------------
    // Parallel positional read
    // -----------------------------------------------------------------------
    bool parallel_read(char* dst, size_t size, size_t file_offset) {
        const size_t hw_threads = static_cast<size_t>(std::thread::hardware_concurrency());
        const size_t max_by_size = size / (1024 * 1024);  // 1 thread per MB
        const size_t num_threads = std::max(size_t{1}, std::min(hw_threads, max_by_size));

        if (num_threads == 1) {
            return single_read(dst, size, file_offset);
        }

        // Align chunk boundaries to page size for natural I/O alignment
        size_t chunk_size = size / num_threads;
        chunk_size = (chunk_size + 4095u) & ~size_t{4095u};

        std::atomic<bool> success{true};
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

#ifdef ENABLE_BD_PROFILING_LOG
        const auto t0 = std::chrono::steady_clock::now();
#endif

        size_t cur_offset = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            const size_t read_size = (i == num_threads - 1u) ? (size - cur_offset) : chunk_size;
            if (read_size == 0) {
                break;
            }

            char* ptr = dst + cur_offset;
            const size_t thread_file_offset = file_offset + cur_offset;

#ifdef _WIN32
            const std::wstring wpath = m_path.native();
            futures.emplace_back(std::async(std::launch::async, [wpath, thread_file_offset, ptr, read_size, &success] {
                HANDLE t_handle = CreateFileW(wpath.c_str(),
                                              GENERIC_READ,
                                              FILE_SHARE_READ | FILE_SHARE_WRITE,
                                              nullptr,
                                              OPEN_EXISTING,
                                              FILE_ATTRIBUTE_NORMAL,
                                              nullptr);
                if (t_handle == INVALID_HANDLE_VALUE) {
                    success = false;
                    return;
                }

                char* cur = ptr;
                size_t remaining = read_size;
                size_t cur_file_offset = thread_file_offset;

                while (remaining > 0 && success) {
                    const DWORD to_read =
                        static_cast<DWORD>(std::min(remaining, static_cast<size_t>(UINT_MAX - 1024u)));
                    OVERLAPPED ov = {};
                    ov.Offset = static_cast<DWORD>(cur_file_offset & 0xFFFFFFFFu);
                    ov.OffsetHigh = static_cast<DWORD>((cur_file_offset >> 32) & 0xFFFFFFFFu);
                    DWORD bytes_read = 0;
                    if (!ReadFile(t_handle, cur, to_read, &bytes_read, &ov)) {
                        if (GetLastError() != ERROR_IO_PENDING) {
                            success = false;
                            break;
                        }
                    }
                    if (bytes_read == 0) {
                        success = false;
                        break;
                    }
                    cur += bytes_read;
                    cur_file_offset += bytes_read;
                    remaining -= bytes_read;
                }
                CloseHandle(t_handle);
            }));
#else
            const int fd = m_fd;
            futures.emplace_back(std::async(std::launch::async, [fd, thread_file_offset, ptr, read_size, &success] {
                char* cur = ptr;
                size_t remaining = read_size;
                off_t cur_file_offset = static_cast<off_t>(thread_file_offset);
                while (remaining > 0 && success) {
                    const ssize_t n = ::pread(fd, cur, remaining, cur_file_offset);
                    if (n <= 0) {
                        success = false;
                        break;
                    }
                    cur += n;
                    cur_file_offset += n;
                    remaining -= static_cast<size_t>(n);
                }
            }));
#endif
            cur_offset += read_size;
        }

        for (auto& f : futures) {
            f.get();
        }
#ifdef ENABLE_BD_PROFILING_LOG
        {
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            const double bw_gbs =
                (elapsed_s > 0.0) ? (static_cast<double>(size) / elapsed_s / (1024.0 * 1024.0 * 1024.0)) : 0.0;
            std::cout << "[ParallelReadStreamBuf] parallel_read: " << size / 1024.0 / 1024.0 << " MB, " << num_threads
                      << " threads, " << elapsed_s * 1e3 << " ms, " << bw_gbs << " GB/s" << std::endl;
        }
#endif
        return success.load();
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    static constexpr size_t UNDERFLOW_BUF = 8192;  // batch size for char-by-char reads

    std::filesystem::path m_path;
#ifdef _WIN32
    HANDLE m_handle = INVALID_HANDLE_VALUE;
#else
    int m_fd = -1;
#endif
    std::streamoff m_header_offset;
    std::streamoff m_file_offset;
    std::streamoff m_file_size = 0;
    size_t m_threshold;
    std::array<char_type, UNDERFLOW_BUF> m_underflow_buf{};  // buffer for underflow()
};

}  // namespace util
}  // namespace ov
