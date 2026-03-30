// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_mem_streambuf.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#ifdef _WIN32
// clang-format off
#    include <windows.h>
#    include <psapi.h>
// clang-format on
#else
#    include <sys/mman.h>

#    include <fstream>
#    include <sstream>
#endif

namespace {

#ifdef _WIN32
/**
 * @brief Convert a kernel device path (\Device\HarddiskVolume3\foo\bar) to a
 *        Win32 drive path (C:\foo\bar).
 */
static bool resolve_device_path(const wchar_t* dev_path, wchar_t* out, DWORD out_len) {
    const size_t MAX_DRIVES_LEN = 512;
    wchar_t drives[MAX_DRIVES_LEN] = {};
    if (!GetLogicalDriveStringsW(MAX_DRIVES_LEN, drives))
        return false;
    for (const wchar_t* d = drives; *d; d += wcslen(d) + 1) {
        wchar_t drive[3] = {d[0], d[1], L'\0'};
        wchar_t dev_name[MAX_PATH] = {};
        if (!QueryDosDeviceW(drive, dev_name, MAX_PATH))
            continue;
        const size_t dev_name_len = wcslen(dev_name);
        if (wcsncmp(dev_path, dev_name, dev_name_len) == 0 &&
            (dev_path[dev_name_len] == L'\\' || dev_path[dev_name_len] == L'\0')) {
            swprintf_s(out, out_len, L"%s%s", drive, dev_path + dev_name_len);
            return true;
        }
    }
    return false;
}
#else
/**
 * @brief Parse /proc/self/maps to find the file backing an mmap address.
 *
 * Returns true and fills out_path/out_offset if the address is inside
 * a named file mapping (not anonymous / [stack] / [heap]).
 */
static bool get_mmap_file_info(const void* addr, std::filesystem::path& out_path, std::streamoff& out_offset) {
    std::ifstream maps_file("/proc/self/maps");
    if (!maps_file.is_open())
        return false;
    const auto addr_val = reinterpret_cast<uintptr_t>(addr);
    std::string line;
    while (std::getline(maps_file, line)) {
        // Format: start-end perms offset dev inode [pathname]
        std::istringstream iss(line);
        std::string addr_range, perms, offset_str, dev, inode_str;
        if (!(iss >> addr_range >> perms >> offset_str >> dev >> inode_str))
            continue;
        const auto dash = addr_range.find('-');
        if (dash == std::string::npos)
            continue;
        uintptr_t range_start = 0, range_end = 0;
        try {
            range_start = static_cast<uintptr_t>(std::stoull(addr_range.substr(0, dash), nullptr, 16));
            range_end = static_cast<uintptr_t>(std::stoull(addr_range.substr(dash + 1), nullptr, 16));
        } catch (...) {
            continue;
        }
        if (addr_val < range_start || addr_val >= range_end)
            continue;
        std::string path;
        if (!(iss >> path) || path.empty() || path[0] != '/')
            return false;  // anonymous or special region, no benefit
        out_path = path;
        std::streamoff map_offset = 0;
        try {
            map_offset = static_cast<std::streamoff>(std::stoull(offset_str, nullptr, 16));
        } catch (...) {
            return false;
        }
        out_offset = map_offset + static_cast<std::streamoff>(addr_val - range_start);
        return true;
    }
    return false;
}
#endif

}  // namespace

namespace ov::util {

ParallelMemStreamBuf::ParallelMemStreamBuf(const void* data, size_t size, size_t threshold)
    : m_begin(static_cast<const char*>(data)),
      m_end(static_cast<const char*>(data) + size),
      m_current(static_cast<const char*>(data)),
      m_threshold(threshold) {
#ifdef _WIN32
    // On Windows, detect whether this memory is a file-backed mmap region.
    // If so, build a ParallelReadStreamBuf over the same file+offset so
    // ReadFile is used instead of mmap+memcpy.  This avoids the 2x RAM
    // pressure (mmap working-set + destination buffer) that causes
    // catastrophic working-set thrashing for multi-GB models, and
    // eliminates per-page PFN database lock contention.
    if (size >= threshold) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(data, &mbi, sizeof(mbi)) && mbi.Type == MEM_MAPPED) {
            wchar_t dev_path[MAX_PATH] = {};
            if (GetMappedFileNameW(GetCurrentProcess(), const_cast<void*>(data), dev_path, MAX_PATH) > 0) {
                wchar_t win32_path[MAX_PATH] = {};
                if (resolve_device_path(dev_path, win32_path, MAX_PATH)) {
                    // Compute file offset: AllocationBase is the start of the mapped view.
                    const std::streamoff file_offset =
                        reinterpret_cast<const char*>(data) - reinterpret_cast<const char*>(mbi.AllocationBase);
                    try {
                        m_file_buf = std::make_unique<ParallelReadStreamBuf>(std::filesystem::path(win32_path),
                                                                             file_offset,
                                                                             threshold);
                    } catch (...) {
                        // File became inaccessible after mmap detection; fall through to memcpy path.
                    }
                }
            }
        }
    }
    // Fallback: issue an upfront async prefetch for the entire region so pages
    // start arriving while the blob header is being parsed.
    if (!m_file_buf) {
        WIN32_MEMORY_RANGE_ENTRY prefetch_range{const_cast<void*>(data), size};
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &prefetch_range, 0);
    }
#else
    // On Linux, detect file-backed mmap via /proc/self/maps.
    // If the pointer falls inside a file mapping, build a ParallelReadStreamBuf
    // (pread-based) to avoid mmap Working Set residency pressure and page-fault
    // overhead that degrades throughput on multi-GB models.
    if (size >= threshold) {
        std::filesystem::path file_path;
        std::streamoff file_off = 0;
        if (get_mmap_file_info(data, file_path, file_off)) {
            try {
                m_file_buf = std::make_unique<ParallelReadStreamBuf>(file_path, file_off, threshold);
            } catch (...) {
                // File became inaccessible after mmap detection; fall through to memcpy path.
            }
        }
    }
    // For non-file-backed memory (anonymous mmap, USM host buffers, etc.)
    // fall back to async prefetch + parallel memcpy.
    if (!m_file_buf) {
        // madvise(2) requires addr to be page-aligned; round down and extend
        // the length to cover the alignment delta.
        const uintptr_t base = reinterpret_cast<uintptr_t>(data);
        const uintptr_t aligned_base = base & ~uintptr_t{4095};
        madvise(reinterpret_cast<void*>(aligned_base), size + (base - aligned_base), MADV_WILLNEED);
    }
#endif
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

ParallelMemStreamBuf::pos_type ParallelMemStreamBuf::seekoff(off_type off,
                                                             std::ios_base::seekdir way,
                                                             std::ios_base::openmode which) {
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
#ifdef _WIN32
    // On Windows, mmap page faults require acquiring the PFN database lock
    // once per page.  Too many concurrent threads cause severe kernel-level
    // serialization.  Cap at 16 threads so PFN-lock contention is bounded
    // while still saturating NVMe queue depth via PrefetchVirtualMemory.
    constexpr size_t MAX_CHUNKS = 16;
    const size_t num_chunks = std::max(size_t{1}, std::min(size / MIN_CHUNK, MAX_CHUNKS));
#else
    // Cap at hardware_concurrency: without a bound, a large anonymous buffer
    // (e.g. 10 GB loaded into a vector) would spawn size/MIN_CHUNK = 5120
    // threads, exhausting pthread stacks and causing severe OS scheduling
    // overhead.  Match the same hw_threads ceiling used by parallel_read.
    const size_t hw_conc = std::max(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t num_chunks = std::max(size_t{1}, std::min(size / MIN_CHUNK, hw_conc));
#endif
    const size_t chunk_size = (size + num_chunks - 1) / num_chunks;

#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY prefetch_range{const_cast<char*>(src), size};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &prefetch_range, 0);
#else
    // Ask the kernel to start async I/O for these mmap pages so they are
    // resident before the parallel memcpy threads access them.
    // madvise(2) requires a page-aligned address; round down and extend size.
    const uintptr_t src_base = reinterpret_cast<uintptr_t>(src);
    const uintptr_t src_aligned = src_base & ~uintptr_t{4095};
    madvise(reinterpret_cast<void*>(src_aligned), size + (src_base - src_aligned), MADV_WILLNEED);
#endif

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

}  // namespace ov::util
