// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parallel_mem_streambuf.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// clang-format off
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <cwchar>
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#    include <sys/mman.h>
#endif

#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu {

#ifdef _WIN32
// Convert a kernel device path (\Device\HarddiskVolume3\foo\bar) to a
// Win32 drive path (C:\foo\bar).
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
        if (wcsncmp(dev_path, dev_name, dev_name_len) == 0 && (dev_path[dev_name_len] == L'\\' || dev_path[dev_name_len] == L'\0')) {
            swprintf_s(out, out_len, L"%s%s", drive, dev_path + dev_name_len);
            return true;
        }
    }
    return false;
}
#endif

// Detect whether a memory address is backed by a file-based mmap.
// On Linux, parses /proc/self/maps.
// On Windows, uses VirtualQuery + GetMappedFileNameW + drive letter resolution.
// param addr        The memory address to inspect.
// param out_path    [out] Path to the backing file (only set on success).
// param out_offset  [out] Absolute byte offset within the file corresponding to addr.
// return true if the address is file-backed, false otherwise.
static bool get_mmap_file_info(const void* addr, std::filesystem::path& out_path, std::streamoff& out_offset) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(addr, &mbi, sizeof(mbi)) || mbi.Type != MEM_MAPPED) {
        return false;
    }
    void* query_addr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(addr));
    wchar_t dev_path[MAX_PATH] = {};
    if (GetMappedFileNameW(GetCurrentProcess(), query_addr, dev_path, MAX_PATH) == 0) {
        return false;
    }
    wchar_t win32_path[MAX_PATH] = {};
    if (!resolve_device_path(dev_path, win32_path, MAX_PATH)) {
        return false;
    }
    out_path = std::filesystem::path(win32_path);
    out_offset = reinterpret_cast<const char*>(addr) - reinterpret_cast<const char*>(mbi.AllocationBase);
    return true;
#else
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
        // Skip whitespace after inode to reach the optional pathname field.
        // Use getline instead of operator>> to handle paths that contain spaces.
        std::string pathname;
        std::getline(iss >> std::ws, pathname);
        if (pathname.empty())
            return false;  // anonymous mapping (no pathname)
        std::filesystem::path path(pathname);
        if (!path.is_absolute())
            return false;  // special region like [heap], [stack], [vdso]
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
#endif
}

// Issue an asynchronous prefetch hint for a memory region.
// On Linux, calls madvise(MADV_WILLNEED) (with page-aligned address).
// On Windows, calls PrefetchVirtualMemory.
// addr  Start of the memory region.
// size  Size of the region in bytes.
static void prefetch_memory(const void* addr, size_t size) {
#ifdef _WIN32
    void* prefetch_addr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(addr));
    WIN32_MEMORY_RANGE_ENTRY prefetch_range{prefetch_addr, size};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &prefetch_range, 0);
#else
    // madvise(2) requires addr to be page-aligned; round down and extend size.
    const uintptr_t base = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t aligned_base = base & ~uintptr_t{4095};
    // madvise is advisory — failure is non-fatal; the read path will still work.
    (void)madvise(reinterpret_cast<void*>(aligned_base), size + (base - aligned_base), MADV_WILLNEED);
#endif
}

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
        if (get_mmap_file_info(data, file_path, file_off)) {
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
        prefetch_memory(data, size);
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
    const size_t num_chunks = std::max(size_t{1}, std::min(size / ov::util::default_parallel_io_min_chunk, hw_conc));
    const size_t chunk_size = (size + num_chunks - 1) / num_chunks;
    prefetch_memory(src, size);

    std::vector<std::thread> workers;
    workers.reserve(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        try {
            workers.emplace_back([&, i]() {
                const size_t offset = i * chunk_size;
                const size_t copy_size = (i + 1 == num_chunks) ? (size - offset) : std::min(chunk_size, size - offset);
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
