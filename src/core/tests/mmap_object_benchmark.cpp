// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/util/mmap_object.hpp"

#ifdef __linux__
#    include <fcntl.h>
#endif

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov::test {

static std::filesystem::path generate_test_file(size_t size_bytes) {
    auto path = std::filesystem::path("test_file" + std::to_string(size_bytes / (1024 * 1024)) + "mb.bin");

    if (std::filesystem::exists(path)) {
        std::cout << "Test file already exists: " << path << ", skipping generation." << std::endl;
        return path;
    }
    std::vector<uint8_t> chunk(1024 * 1024);
    for (size_t i = 0; i < chunk.size(); ++i)
        chunk[i] = static_cast<uint8_t>(i % 251);
    std::ofstream f(path, std::ios::binary);
    for (size_t written = 0; written < size_bytes; written += chunk.size()) {
        const auto to_write = std::min(chunk.size(), size_bytes - written);
        f.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(to_write));
    }
    return path;
}

static long long measure_ms(const std::function<void()>& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

static void evict_cache(const std::filesystem::path& path, size_t file_size) {
    // Prefer /proc/sys/vm/drop_caches (requires root / CAP_SYS_ADMIN, available when the container
    // is started with --privileged).  Writing "3" flushes the host page
    // cache, dentries, and inodes — the only fully reliable way to guarantee a cold-cache run.
    // Fallback: posix_fadvise(DONTNEED) is best-effort; the kernel may ignore it.
    static const bool have_drop_caches = [] {
        int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
        if (fd >= 0) {
            ::close(fd);
            return true;
        }
        return false;
    }();

    if (have_drop_caches) {
        ::sync();  // commit all dirty pages before dropping
        int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
        if (fd >= 0) {
            std::ignore = ::write(fd, "3", 1);
            ::close(fd);
        }
        sleep(1);  // give the kernel a moment to settle
        return;
    } else {
        std::cout << "No access to /proc/sys/vm/drop_caches, falling back to posix_fadvise(DONTNEED)" << std::endl;
    }

    // Fallback: best-effort fadvise.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        std::cout << "posix_fadvise(DONTNEED) in use, kernel might ignore it" << std::endl;
        posix_fadvise(fd, 0, static_cast<off_t>(file_size), POSIX_FADV_DONTNEED);
        ::close(fd);
    }
}

static long long bench(const std::function<void()>& fn,
                       const std::filesystem::path& path,
                       size_t file_size,
                       int warmup_runs = 1,
                       int measured_runs = 5) {
    for (int i = 0; i < warmup_runs; ++i) {
        evict_cache(path, file_size);
        fn();
        evict_cache(path, file_size);
    }
    long long total = 0;
    for (int i = 0; i < measured_runs; ++i) {
        evict_cache(path, file_size);
        total += measure_ms(fn);
    }
    return total / measured_runs;
}

static double throughput_mbs(size_t size_mb, long long ms) {
    if (ms <= 0)
        return 0.0;
    return static_cast<double>(size_mb) * 1000.0 / static_cast<double>(ms);
}

static void strategy_hint_prefetch(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch();
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1]; //prevents optimization
    }
}

static void strategy_no_prefault(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1]; //prevents optimization
    }
}

static void strategy_ifstream_read(const std::filesystem::path& path, size_t file_size) {
    std::vector<char> read_buffer(file_size);
    std::ifstream f(path, std::ios::binary);
    f.read(read_buffer.data(), static_cast<std::streamsize>(file_size));
    volatile char sink = read_buffer[0] + read_buffer[file_size / 2] + read_buffer[file_size - 1];
    (void)sink;
}

// Strategy: mmap + hint_prefetch only — measures time to make all data accessible
// without any copy overhead. Stop the clock when hint_prefetch returns; data is
// then immediately accessible via the mapping at zero extra cost.
static void strategy_prefetch_only_dos(const std::filesystem::path& path) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch();
    (void)mapped;
}

// Strategy: ifstream into a single pre-allocated buffer — one kernel→user copy,
// no secondary memcpy. This is the minimum cost of ifstream-based loading and the
// fair counterpart to strategy_prefetch_only_dos: ifstream always copies; mmap does not.
static void strategy_ifstream_into_buffer_dos(const std::filesystem::path& path, size_t file_size) {
    std::vector<char> buffer(file_size);
    std::ifstream f(path, std::ios::binary);
    f.read(buffer.data(), static_cast<std::streamsize>(file_size));
    volatile char sink = buffer[0] + buffer[file_size / 2] + buffer[file_size - 1];
    (void)sink;
}

// Strategy: mmap + single first access — measures cold-cache latency to the first
// page fault with no prefetch at all. The kernel loads exactly one 4 KiB page on
// the first touch; this is the minimum latency before the caller can read any byte.
static void strategy_first_access_dos(const std::filesystem::path& path) {
    auto mapped = load_mmap_object(path);
    volatile char sink = mapped->data()[0];
    (void)sink;
}

#ifdef __linux__
// Strategy: O_DIRECT read — bypasses the page cache entirely, measuring raw NVMe
// sequential throughput. Use as a sanity-check reference: if any other strategy
// reports higher throughput, the page cache was not fully evicted before that run.
static void strategy_direct_read_dos(const std::filesystem::path& path, size_t file_size) {
    constexpr size_t align = 4096;
    constexpr size_t buf_size = 128 * 1024 * 1024;  // 128 MiB, sector-aligned
    void* raw = aligned_alloc(align, buf_size);
    if (!raw)
        return;
    char* buf = static_cast<char*>(raw);
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        std::free(raw);
        return;
    }
    volatile char sink = 0;
    size_t remaining = file_size;
    while (remaining > 0) {
        // O_DIRECT requires transfer size to be a multiple of the block size (align to 4096).
        const size_t to_read = std::min(buf_size, (remaining / align) * align);
        if (to_read == 0)
            break;
        const ssize_t n = read(fd, buf, to_read);
        if (n <= 0)
            break;
        const auto nr = static_cast<size_t>(n);
        sink += buf[0] + buf[nr / 2] + buf[nr - 1];
        remaining -= nr;
    }
    close(fd);
    std::free(raw);
}
#endif

static void strategy_hint_prefetch_partial(const std::filesystem::path& path,
                                           size_t /*file_size*/,
                                           size_t offset,
                                           size_t size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch(offset, size);
    const auto total_copy_size = std::min(size, mapped->size() - offset);
    // Copy in chunks to support large regions
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> buffer(std::min(chunk_size, total_copy_size));
    volatile char sink = 0;
    for (size_t i = 0; i < total_copy_size; i += chunk_size) {
        const size_t copy_size = std::min(chunk_size, total_copy_size - i);
        std::memcpy(buffer.data(), mapped->data() + offset + i, copy_size);
        // Sample multiple positions to prevent memcpy optimization
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];
    }
}

// ─── Benchmark Tests ────────────────────────────────────────────────────────
// All tests are DISABLED_ so they won't run in CI.
// Run manually with: --gtest_also_run_disabled_tests --gtest_filter=*MmapBenchmark*

class MmapBenchmark : public ::testing::Test {};

TEST_F(MmapBenchmark, DISABLED_latency_and_throughput_table) {
    const std::vector<size_t> sizes_mb = {10, 100, 500, 1000};
    constexpr int warmup = 0;
    constexpr int runs = 3;

    // Generate all test files
    struct TestFile {
        size_t size_mb;
        size_t size_bytes;
        std::filesystem::path path;
    };
    std::vector<TestFile> files;
    for (size_t mb : sizes_mb) {
        TestFile tf;
        tf.size_mb = mb;
        tf.size_bytes = mb * 1024 * 1024;
        tf.path = generate_test_file(tf.size_bytes);
        evict_cache(tf.path, tf.size_bytes);  // Flush dirty pages from file generation
        files.push_back(tf);
    }

    // Collect results: [file_idx] -> {hint_prefetch, no_prefault, ifstream}
    struct Row {
        size_t mb;
        long long t_hint_prefetch;
        long long t_no_prefault;
        long long t_ifstream;
    };
    std::vector<Row> results;

    for (const auto& tf : files) {
        Row r{};
        r.mb = tf.size_mb;
        r.t_ifstream = bench(
            [&]() {
                strategy_ifstream_read(tf.path, tf.size_bytes);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
        r.t_no_prefault = bench(
            [&]() {
                strategy_no_prefault(tf.path, tf.size_bytes);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
        r.t_hint_prefetch = bench(
            [&]() {
                strategy_hint_prefetch(tf.path, tf.size_bytes);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
        results.push_back(r);
    }

    // Print latency table
    printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
    printf("%-10s | %17s | %13s | %13s\n", "Size (MB)", "hint_prefetch", "no-prefault", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %14lld ms | %10lld ms | %10lld ms\n", r.mb, r.t_hint_prefetch, r.t_no_prefault, r.t_ifstream);
    }

    // Print throughput table
    printf("\n--- Throughput (MB/s) ---\n");
    printf("%-10s | %17s | %13s | %13s\n", "Size (MB)", "hint_prefetch", "no-prefault", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %12.0f MB/s | %8.0f MB/s | %8.0f MB/s\n",
               r.mb,
               throughput_mbs(r.mb, r.t_hint_prefetch),
               throughput_mbs(r.mb, r.t_no_prefault),
               throughput_mbs(r.mb, r.t_ifstream));
    }
}

TEST_F(MmapBenchmark, DISABLED_load_latency_comparison) {
    // Measures "time from cold cache to data ready to use"
    //
    //  prefetch_only     : mmap + hint_prefetch (no copy). Data is ready when call returns.
    //  ifstream_into_buf : ifstream::read into one heap buffer (one unavoidable kernel copy).
    //  first_access      : mmap + touch data()[0] only. Baseline: one cold page fault, no prefetch.
    //  direct_read       : O_DIRECT, bypasses page cache. Raw NVMe reference.
    //                      If another strategy exceeds this throughput, eviction was incomplete.
    const std::vector<size_t> sizes_mb = {10, 100, 500, 1000};
    constexpr int warmup = 1;
    constexpr int runs = 5;

    struct TestFile {
        size_t size_mb;
        size_t size_bytes;
        std::filesystem::path path;
    };
    std::vector<TestFile> files;
    for (size_t mb : sizes_mb) {
        TestFile tf;
        tf.size_mb = mb;
        tf.size_bytes = mb * 1024 * 1024;
        tf.path = generate_test_file(tf.size_bytes);
        evict_cache(tf.path, tf.size_bytes);
        files.push_back(tf);
    }

    struct Row {
        size_t mb;
        long long t_prefetch_only;
        long long t_ifstream_buf;
        long long t_first_access;
#ifdef __linux__
        long long t_direct_read;
#endif
    };
    std::vector<Row> results;

    for (const auto& tf : files) {
        Row r{};
        r.mb = tf.size_mb;
        r.t_prefetch_only = bench(
            [&]() {
                strategy_prefetch_only_dos(tf.path);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
        r.t_ifstream_buf = bench(
            [&]() {
                strategy_ifstream_into_buffer_dos(tf.path, tf.size_bytes);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
        r.t_first_access = bench(
            [&]() {
                strategy_first_access_dos(tf.path);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
#ifdef __linux__
        r.t_direct_read = bench(
            [&]() {
                strategy_direct_read_dos(tf.path, tf.size_bytes);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
#endif
        results.push_back(r);
    }

    // Latency table
    printf("\n--- Load latency: time to data-ready (ms, mean of %d runs, cold cache) ---\n", runs);
#ifdef __linux__
    printf("%-10s | %16s | %17s | %14s | %13s\n",
           "Size (MB)",
           "prefetch_only",
           "ifstream_into_buf",
           "first_access",
           "direct_read");
    printf("%-10s-|-%16s-|-%17s-|-%14s-|-%13s\n",
           "----------",
           "----------------",
           "-----------------",
           "--------------",
           "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %13lld ms | %14lld ms | %11lld ms | %10lld ms\n",
               r.mb,
               r.t_prefetch_only,
               r.t_ifstream_buf,
               r.t_first_access,
               r.t_direct_read);
    }
#else
    printf("%-10s | %16s | %17s | %14s\n", "Size (MB)", "prefetch_only", "ifstream_into_buf", "first_access");
    printf("%-10s-|-%16s-|-%17s-|-%14s\n", "----------", "----------------", "-----------------", "--------------");
    for (const auto& r : results) {
        printf("%-10zu | %13lld ms | %14lld ms | %11lld ms\n",
               r.mb,
               r.t_prefetch_only,
               r.t_ifstream_buf,
               r.t_first_access);
    }
#endif

    // Throughput table
    printf("\n--- Throughput (MB/s) ---\n");
#ifdef __linux__
    printf("%-10s | %16s | %17s | %14s | %13s\n",
           "Size (MB)",
           "prefetch_only",
           "ifstream_into_buf",
           "first_access",
           "direct_read");
    printf("%-10s-|-%16s-|-%17s-|-%14s-|-%13s\n",
           "----------",
           "----------------",
           "-----------------",
           "--------------",
           "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %11.0f MB/s | %12.0f MB/s | %9.0f MB/s | %8.0f MB/s\n",
               r.mb,
               throughput_mbs(r.mb, r.t_prefetch_only),
               throughput_mbs(r.mb, r.t_ifstream_buf),
               throughput_mbs(r.mb, r.t_first_access),
               throughput_mbs(r.mb, r.t_direct_read));
    }
#else
    printf("%-10s | %16s | %17s | %14s\n", "Size (MB)", "prefetch_only", "ifstream_into_buf", "first_access");
    printf("%-10s-|-%16s-|-%17s-|-%14s\n", "----------", "----------------", "-----------------", "--------------");
    for (const auto& r : results) {
        printf("%-10zu | %11.0f MB/s | %12.0f MB/s | %9.0f MB/s\n",
               r.mb,
               throughput_mbs(r.mb, r.t_prefetch_only),
               throughput_mbs(r.mb, r.t_ifstream_buf),
               throughput_mbs(r.mb, r.t_first_access));
    }
#endif
}

TEST_F(MmapBenchmark, DISABLED_partial_prefault_offset_table) {
    constexpr size_t file_size_mb = 1200;
    constexpr size_t file_size = file_size_mb * 1024 * 1024;
    constexpr int warmup = 0;
    constexpr int runs = 3;

    auto file_path = generate_test_file(file_size);
    evict_cache(file_path, file_size);  // Flush dirty pages from file generation

    const std::vector<size_t> offsets_mb = {0, 1, 17, 500, 700, 800};
    const std::vector<size_t> region_sizes_mb = {10, 100, 500};

    // Pre-compute all results[size_idx][offset_idx]; -1 means "exceeds"
    std::vector<std::vector<long long>> results(region_sizes_mb.size(), std::vector<long long>(offsets_mb.size(), -1));
    for (size_t si = 0; si < region_sizes_mb.size(); ++si) {
        for (size_t oi = 0; oi < offsets_mb.size(); ++oi) {
            const size_t off_bytes = offsets_mb[oi] * 1024 * 1024;
            const size_t sz_bytes = region_sizes_mb[si] * 1024 * 1024;
            if (off_bytes + sz_bytes > file_size)
                continue;
            results[si][oi] = bench(
                [&]() {
                    strategy_hint_prefetch_partial(file_path, file_size, off_bytes, sz_bytes);
                },
                file_path,
                file_size,
                warmup,
                runs);
        }
    }

    // Print: sizes as rows, offsets as columns
    printf("\n--- partial prefault: hint_prefetch with offset ---\n");
    printf("  %-14s", "size \\ offset");
    for (size_t off_mb : offsets_mb)
        printf(" | %7zu MB", off_mb);
    printf("\n  %s\n", std::string(14 + offsets_mb.size() * 14, '-').c_str());

    for (size_t si = 0; si < region_sizes_mb.size(); ++si) {
        printf("  %-14zu", region_sizes_mb[si]);
        for (size_t oi = 0; oi < offsets_mb.size(); ++oi) {
            if (results[si][oi] < 0)
                printf(" |    (exceeds)");
            else
                printf(" | %7lld ms", results[si][oi]);
        }
        printf("\n");
    }
}

}  // namespace ov::test
