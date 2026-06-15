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
        static bool warned = false;
        if (!warned) {
            std::cout << "[WARNING] No access to /proc/sys/vm/drop_caches, falling back to "
                         "posix_fadvise(DONTNEED). Results may be unreliable (kernel can ignore the hint)."
                      << std::endl;
            warned = true;
        }
    }

    // Fallback: best-effort fadvise.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
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

// Strategy: mmap + hint_prefetch — measures time to make all data accessible
// without any copy overhead. data is accessible via the mapping at zero extra cost.
static void strategy_hint_prefetch(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch();
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];  // prevents optimization
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
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];  // prevents optimization
    }
}

// Strategy: ifstream into a single pre-allocated buffer — one kernel→user copy
static void strategy_ifstream_read(const std::filesystem::path& path, size_t file_size) {
    std::vector<char> read_buffer(file_size);
    std::ifstream f(path, std::ios::binary);
    f.read(read_buffer.data(), static_cast<std::streamsize>(file_size));
    volatile char sink = read_buffer[0] + read_buffer[file_size / 2] + read_buffer[file_size - 1];
    (void)sink;
}

static void strategy_hint_prefetch_partial(const std::filesystem::path& path,
                                           size_t /*file_size*/,
                                           size_t offset,
                                           size_t size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_prefetch(offset, size);
    const auto total_copy_size = std::min(size, mapped->size() - offset);
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

// ─── FileLoadBenchmark ──────────────────────────────────────────────────────
//
// Developer-only benchmarks for comparing file-loading strategies (mmap with
// prefetch, mmap without prefetch, ifstream, O_DIRECT).  Use these to evaluate
// I/O performance on new hardware or to validate changes to ov::MmapObject.
//
// These tests are NOT compiled by default.  To include them in the build:
//
//   cmake -DENABLE_TESTS=ON -DENABLE_DEVELOPER_TESTS=ON <other flags> ..
//
// To run tests:
//   ./ov_core_unit_tests --gtest_filter=*FileLoadBenchmark*
//
// Requirements for reliable results:
//   - Build OpenVINO in Release mode (CMAKE_BUILD_TYPE=Release).
//   - Run inside a privileged container (docker run --privileged) or as root
//     so that /proc/sys/vm/drop_caches is writable.  Without it the benchmark falls back
//     to posix_fadvise(DONTNEED) which the kernel may ignore.
//

class FileLoadBenchmark : public ::testing::Test {};

TEST_F(FileLoadBenchmark, strategists_latency_and_throughput_table) {
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
        evict_cache(tf.path, tf.size_bytes);
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

    printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
    printf("%-10s | %17s | %13s | %13s\n", "Size (MB)", "hint_prefetch", "no-prefault", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %14lld ms | %10lld ms | %10lld ms\n", r.mb, r.t_hint_prefetch, r.t_no_prefault, r.t_ifstream);
    }

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

TEST_F(FileLoadBenchmark, hint_prefetch_with_offset_table) {
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
