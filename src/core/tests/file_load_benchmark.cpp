// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

#ifdef __linux__
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#endif

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov::test {

namespace {
#ifdef __linux__
const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
#else
const size_t page_size = 4096;  // Fallback for non-Linux platforms
#endif

struct TestFile {
    size_t size_mib;
    std::filesystem::path path;

    size_t size_bytes() const {return size_mib * 1024 * 1024; }
};

std::filesystem::path generate_test_file(const TestFile& tf) {
    auto path = std::filesystem::path("test_file" + std::to_string(tf.size_mib) + "mib.bin");

    if (util::file_exists(path) && std::filesystem::file_size(path) == tf.size_bytes()) {
        std::cout << "Test file already exists: " << path << ", skipping generation." << std::endl;
        return path;
    }
    std::vector<uint8_t> chunk(1024 * 1024);
    for (size_t i = 0; i < chunk.size(); ++i)
        chunk[i] = static_cast<uint8_t>(i % 251);
    std::ofstream f(path, std::ios::binary);
    for (size_t written = 0; written < tf.size_bytes(); written += chunk.size()) {
        const auto to_write = std::min(chunk.size(), tf.size_bytes() - written);
        f.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(to_write));
    }
    return path;
}

long long measure_ms(const std::function<void()>& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

void evict_cache(const std::filesystem::path& path, size_t file_size) {
    static bool warned = false;
    auto warn_once = [](const char* msg) {
        if (!warned) {
            std::cout << "[WARNING] " << msg << " Results may be unreliable." << std::endl;
            warned = true;
        }
    };

#ifdef __linux__
    // Prefer /proc/sys/vm/drop_caches (requires root / CAP_SYS_ADMIN, available when the container
    // is started with --privileged).  Writing "3" flushes the host page
    // cache, dentries, and inodes — the only fully reliable way to guarantee a cold-cache run.
    // Fallback: posix_fadvise(DONTNEED) is best-effort; the kernel may ignore it.
    ::sync();  // commit all dirty pages before dropping
    if (std::ofstream drop_caches("/proc/sys/vm/drop_caches"); drop_caches) {
        drop_caches << "3";
        sleep(1);  // give the kernel a moment to settle
        return;
    }

    warn_once("No access to /proc/sys/vm/drop_caches, falling back to posix_fadvise(DONTNEED).");

    // Fallback: best-effort fadvise.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, static_cast<off_t>(file_size), POSIX_FADV_DONTNEED);
        ::close(fd);
    }
#elif defined(_WIN32)
    // Windows moves evicted file pages to the standby list ("Cached" in Task Manager). Purging it is
    // the equivalent of `drop_caches 1`: NtSetSystemInformation(SystemMemoryListInformation,
    // MemoryPurgeStandbyList) clears it. Requires SeProfileSingleProcessPrivilege (run elevated).
    // Both mmap and read paths start cold once the standby list is empty.
    (void)path;
    (void)file_size;

    enum SYSTEM_MEMORY_LIST_COMMAND { MemoryPurgeStandbyList = 4 };
    constexpr int SystemMemoryListInformation = 80;
    using NtSetSystemInformation_t = LONG(WINAPI*)(int, PVOID, ULONG);

    // Acquire the privilege required to purge the standby list.
    HANDLE token = nullptr;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &token)) {
        TOKEN_PRIVILEGES tp{};
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        LookupPrivilegeValue(nullptr, SE_PROF_SINGLE_PROCESS_NAME, &tp.Privileges[0].Luid);
        AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), nullptr, nullptr);
        CloseHandle(token);
    }

    auto ntdll = GetModuleHandleW(L"ntdll.dll");
    auto nt_set = ntdll ? reinterpret_cast<NtSetSystemInformation_t>(
                              GetProcAddress(ntdll, "NtSetSystemInformation"))
                        : nullptr;
    if (!nt_set) {
        warn_once("NtSetSystemInformation unavailable; cannot purge standby list.");
        return;
    }
    int command = MemoryPurgeStandbyList;
    if (nt_set(SystemMemoryListInformation, &command, sizeof(command)) != 0) {
        warn_once("Standby-list purge failed (run elevated for cold-cache benchmarking).");
    }
#else
    (void)path;
    (void)file_size;
    warn_once("No cache eviction strategy available on this platform.");
#endif
}

long long bench(const std::function<void()>& fn,
                const std::filesystem::path& path,
                size_t file_size,
                int warmup_runs = 1,
                int measured_runs = 5) {
    for (int i = 0; i < warmup_runs; ++i) {
        evict_cache(path, file_size);
        fn();
    }
    long long total = 0;
    for (int i = 0; i < measured_runs; ++i) {
        evict_cache(path, file_size);
        total += measure_ms(fn);
    }
    return total / measured_runs;
}

double throughput_mibs(size_t size_mib, long long ms) {
    if (ms <= 0)
        return 0.0;
    return static_cast<double>(size_mib) * 1000.0 / static_cast<double>(ms);
}

namespace strategy {

#ifdef __linux__
// mlock forces every page resident before returning; munlock releases the pin without evicting.
// Bounded by RLIMIT_MEMLOCK -- no limit on a privileged process.
void mlock_munlock(const std::shared_ptr<ov::MappedMemory>& mapped) {
    ASSERT_EQ(::mlock(mapped->data(), mapped->size()), 0)
        << "mlock failed (errno=" << errno << "); check RLIMIT_MEMLOCK";
    ::munlock(mapped->data(), mapped->size());
}

// Note: the mmap destructor (munmap + close) runs inside the timed window;
void mmap_prefetch_mlock(const std::filesystem::path& path, size_t /*file_size*/) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch();  // synchronous for regions > 4 MiB (parallel touch + join)
    mlock_munlock(mapped);    // should be near no-op and just lock/unlock resident pages
}

void mmap_touch_mlock(const std::filesystem::path& path, size_t /*file_size*/) {
    auto mapped = load_mmap_object(path);
    volatile uint8_t sink = 0;
    for (auto first = mapped->data(), last = first + mapped->size(); first < last; first += page_size) {
        sink += *first;
    }
    mlock_munlock(mapped);  // should be near no-op and just lock/unlock resident pages
}
#endif

void mmap_prefetch_then_memcpy(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch();
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MiB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];  // prevents optimization
    }
}

void mmap_then_memcpy(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MiB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];  // prevents optimization
    }
}

// Strategy: ifstream into a single pre-allocated buffer — one kernel→user copy
void ifstream_read(const std::filesystem::path& path, size_t file_size) {
    std::vector<char> read_buffer(file_size);
    std::ifstream f(path, std::ios::binary);
    f.read(read_buffer.data(), static_cast<std::streamsize>(file_size));
    volatile char sink = read_buffer[0] + read_buffer[file_size / 2] + read_buffer[file_size - 1];
    (void)sink;
}

void mmap_prefetch_then_memcpy_partial(const std::filesystem::path& path,
                                       size_t /*file_size*/,
                                       size_t offset,
                                       size_t size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch(offset, size);
    const auto total_copy_size = std::min(size, mapped->size() - offset);
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MiB chunks
    std::vector<char> buffer(std::min(chunk_size, total_copy_size));
    volatile char sink = 0;
    for (size_t i = 0; i < total_copy_size; i += chunk_size) {
        const size_t copy_size = std::min(chunk_size, total_copy_size - i);
        std::memcpy(buffer.data(), mapped->data() + offset + i, copy_size);
        // Sample multiple positions to prevent memcpy optimization
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];
    }
}

}  // namespace strategy

}  // namespace

// See developer_benchmarks.md for build/run instructions.

class FileLoadBenchmark : public ::testing::Test {};

TEST_F(FileLoadBenchmark, strategies_read_memcpy) {        
    const std::vector<size_t> sizes_mib = {10, 100, 500, 1000};
    constexpr int warmup = 0;
    constexpr int runs = 3;

    // Generate all test files
    std::vector<TestFile> files;
    for (size_t mib : sizes_mib) {
        TestFile tf{mib, {}};
        tf.path = generate_test_file(tf);
        evict_cache(tf.path, tf.size_bytes());
        files.push_back(tf);
    }

    // Collect results: [file_idx] -> {mmap_prefetch_memcpy, mmap_memcpy, ifstream}
    struct Row {
        size_t mib;
        long long t_hint_prefetch;
        long long t_no_prefault;
        long long t_ifstream;
    };
    std::vector<Row> results;

    for (const auto& tf : files) {
        Row r{};
        r.mib = tf.size_mib;
        r.t_ifstream = bench(
            [&]() {
                strategy::ifstream_read(tf.path, tf.size_bytes());
            },
            tf.path,
            tf.size_bytes(),
            warmup,
            runs);
        r.t_no_prefault = bench(
            [&]() {
                strategy::mmap_then_memcpy(tf.path, tf.size_bytes());
            },
            tf.path,
            tf.size_bytes(),
            warmup,
            runs);
        r.t_hint_prefetch = bench(
            [&]() {
                strategy::mmap_prefetch_then_memcpy(tf.path, tf.size_bytes());
            },
            tf.path,
            tf.size_bytes(),
            warmup,
            runs);
        results.push_back(r);
    }

    printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
    printf("%-10s | %17s | %13s | %13s\n", "Size (MiB)", "prefetch+memcpy", "mmap+memcpy", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %14lld ms | %10lld ms | %10lld ms\n", r.mib, r.t_hint_prefetch, r.t_no_prefault, r.t_ifstream);
    }

    printf("\n--- Throughput (MiB/s) ---\n");
    printf("%-10s | %17s | %13s | %13s\n", "Size (MiB)", "prefetch+memcpy", "mmap+memcpy", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %12.0f MiB/s | %8.0f MiB/s | %8.0f MiB/s\n",
               r.mib,
               throughput_mibs(r.mib, r.t_hint_prefetch),
               throughput_mibs(r.mib, r.t_no_prefault),
               throughput_mibs(r.mib, r.t_ifstream));
    }
}

#ifdef __linux__
TEST_F(FileLoadBenchmark, strategies_mlock) {
    const std::vector<size_t> sizes_mib = {10, 100, 500, 1000};
    constexpr int warmup = 0;
    constexpr int runs = 3;

    // Generate all test files
    std::vector<TestFile> files;
    for (size_t mib : sizes_mib) {
        TestFile tf{mib, {}};
        tf.path = generate_test_file(tf);
        evict_cache(tf.path, tf.size_bytes());
        files.push_back(tf);
    }

    // Collect results: [file_idx] -> {mmap_prefetch_mlock, mmap_touch_mlock}
    struct Row {
        size_t mib;
        long long t_prefetch_mlock;
        long long t_mlock;
    };
    std::vector<Row> results;

    for (const auto& tf : files) {
        Row r{};
        r.mib = tf.size_mib;
        r.t_mlock = bench(
            [&]() {
                strategy::mmap_touch_mlock(tf.path, tf.size_bytes());
            },
            tf.path,
            tf.size_bytes(),
            warmup,
            runs);
        r.t_prefetch_mlock = bench(
            [&]() {
                strategy::mmap_prefetch_mlock(tf.path, tf.size_bytes());
            },
            tf.path,
            tf.size_bytes(),
            warmup,
            runs);
        results.push_back(r);
    }

    printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
    printf("%-10s | %17s | %13s\n", "Size (MiB)", "prefetch+mlock", "mmap+mlock");
    printf("%-10s-|-%17s-|-%13s\n", "----------", "-----------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %14lld ms | %10lld ms\n", r.mib, r.t_prefetch_mlock, r.t_mlock);
    }

    printf("\n--- Throughput (MiB/s) ---\n");
    printf("%-10s | %17s | %13s\n", "Size (MiB)", "prefetch+mlock", "mmap+mlock");
    printf("%-10s-|-%17s-|-%13s\n", "----------", "-----------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %12.0f MiB/s | %8.0f MiB/s\n",
               r.mib,
               throughput_mibs(r.mib, r.t_prefetch_mlock),
               throughput_mibs(r.mib, r.t_mlock));
    }
}

#endif

TEST_F(FileLoadBenchmark, hint_prefetch_with_offset_table) {
    constexpr size_t file_size_mib = 1200;
    constexpr int warmup = 0;
    constexpr int runs = 3;

    TestFile tf{file_size_mib, {}};
    tf.path = generate_test_file(tf);
    evict_cache(tf.path, tf.size_bytes());  // Flush dirty pages from file generation

    const std::vector<size_t> offsets_mib = {0, 1, 17, 500, 700, 800};
    const std::vector<size_t> region_sizes_mib = {10, 100, 500};

    // Pre-compute all results[size_idx][offset_idx]; -1 means "exceeds"
    std::vector<std::vector<long long>> results(region_sizes_mib.size(), std::vector<long long>(offsets_mib.size(), -1));
    for (size_t si = 0; si < region_sizes_mib.size(); ++si) {
        for (size_t oi = 0; oi < offsets_mib.size(); ++oi) {
            const size_t off_bytes = offsets_mib[oi] * 1024 * 1024;
            const size_t sz_bytes = region_sizes_mib[si] * 1024 * 1024;
            if (off_bytes + sz_bytes > tf.size_bytes())
                continue;
            results[si][oi] = bench(
                [&]() {
                    strategy::mmap_prefetch_then_memcpy_partial(tf.path, tf.size_bytes(), off_bytes, sz_bytes);
                },
                tf.path,
                tf.size_bytes(),
                warmup,
                runs);
        }
    }

    // Print: sizes as rows, offsets as columns
    printf("\n--- partial prefault: hint_prefetch with offset ---\n");
    printf("  %-14s", "size \\ offset");
    for (size_t off_mib : offsets_mib)
        printf(" | %7zu MiB", off_mib);
    printf("\n  %s\n", std::string(14 + offsets_mib.size() * 14, '-').c_str());

    for (size_t si = 0; si < region_sizes_mib.size(); ++si) {
        printf("  %-14zu", region_sizes_mib[si]);
        for (size_t oi = 0; oi < offsets_mib.size(); ++oi) {
            if (results[si][oi] < 0)
                printf(" |    (exceeds)");
            else
                printf(" | %7lld ms", results[si][oi]);
        }
        printf("\n");
    }
}

}  // namespace ov::test
