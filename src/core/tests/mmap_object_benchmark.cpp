// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/util/mmap_object.hpp"

#ifdef __linux__
#    include <fcntl.h>
#    include <unistd.h>
#endif

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov::test {

static std::filesystem::path generate_test_file(size_t size_bytes) {
    auto path = std::filesystem::path(utils::generateTestFilePrefix() + "_bench_" +
                                      std::to_string(size_bytes / (1024 * 1024)) + "mb.bin");
    std::vector<char> chunk(1024 * 1024);
    std::iota(chunk.begin(), chunk.end(), char{0});
    std::ofstream f(path, std::ios::binary);
    for (size_t written = 0; written < size_bytes; written += chunk.size()) {
        const auto to_write = std::min(chunk.size(), size_bytes - written);
        f.write(chunk.data(), static_cast<std::streamsize>(to_write));
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
    // #ifdef __linux__
    // posix_fadvise(DONTNEED) evicts pages from page cache (best-effort, kernel may ignore under pressure)
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, static_cast<off_t>(file_size), POSIX_FADV_DONTNEED);
        close(fd);
    }
    // #else
    //     auto mapped = load_mmap_object(path);
    //     mapped->hint_evict();
    //     (void)file_size;
    // #endif
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

// ─── Strategies ─────────────────────────────────────────────────────────────

// Strategy: hint_populate (mmap, parallel prefault + fadvise hints)
static void strategy_hint_populate(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_populate();
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        // Sample multiple positions to prevent memcpy optimization
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];
    }
}

// Strategy: mmap + no prefault (sequential page fault)
static void strategy_no_prefault(const std::filesystem::path& path, size_t file_size) {
    auto mapped = load_mmap_object(path);
    // Copy in chunks to support large files without excessive memory allocation
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> buffer(std::min(chunk_size, file_size));
    volatile char sink = 0;
    for (size_t offset = 0; offset < file_size; offset += chunk_size) {
        const size_t copy_size = std::min(chunk_size, file_size - offset);
        std::memcpy(buffer.data(), mapped->data() + offset, copy_size);
        // Sample multiple positions to prevent memcpy optimization
        sink += buffer[0] + buffer[copy_size / 2] + buffer[copy_size - 1];
    }
}

// Strategy: sequential read via ifstream
static void strategy_ifstream_read(const std::filesystem::path& path, size_t /*file_size*/) {
    std::ifstream f(path, std::ios::binary);
    constexpr size_t chunk_size = 128 * 1024 * 1024;  // 128 MB chunks
    std::vector<char> read_buffer(chunk_size);
    std::vector<char> copy_buffer(chunk_size);
    volatile char sink = 0;
    while (f.read(read_buffer.data(), static_cast<std::streamsize>(read_buffer.size())) || f.gcount() > 0) {
        const auto count = static_cast<size_t>(f.gcount());
        std::memcpy(copy_buffer.data(), read_buffer.data(), count);
        // Sample multiple positions to prevent memcpy optimization
        sink += copy_buffer[0] + copy_buffer[count / 2] + copy_buffer[count - 1];
    }
}

// Strategy: hint_populate with partial region
static void strategy_hint_populate_partial(const std::filesystem::path& path,
                                           size_t /*file_size*/,
                                           size_t offset,
                                           size_t size) {
    auto mapped = load_mmap_object(path);
    mapped->hint_populate(offset, size);
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
    constexpr int warmup = 1;
    constexpr int runs = 5;

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

    // Collect results: [file_idx] -> {hint_populate, no_prefault, ifstream}
    struct Row {
        size_t mb;
        long long t_hint_populate;
        long long t_no_prefault;
        long long t_ifstream;
    };
    std::vector<Row> results;

    for (const auto& tf : files) {
        Row r{};
        r.mb = tf.size_mb;
        r.t_hint_populate = bench(
            [&]() {
                strategy_hint_populate(tf.path, tf.size_bytes);
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
        r.t_ifstream = bench(
            [&]() {
                strategy_ifstream_read(tf.path, tf.size_bytes);
            },
            tf.path,
            tf.size_bytes,
            warmup,
            runs);
        results.push_back(r);
    }

    // Print latency table
    printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
    printf("%-10s | %17s | %13s | %13s\n", "Size (MB)", "hint_populate", "no-prefault", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %14lld ms | %10lld ms | %10lld ms\n", r.mb, r.t_hint_populate, r.t_no_prefault, r.t_ifstream);
    }

    // Print throughput table
    printf("\n--- Throughput (MB/s) ---\n");
    printf("%-10s | %17s | %13s | %13s\n", "Size (MB)", "hint_populate", "no-prefault", "ifstream");
    printf("%-10s-|-%17s-|-%13s-|-%13s\n", "----------", "-----------------", "-------------", "-------------");
    for (const auto& r : results) {
        printf("%-10zu | %12.0f MB/s | %8.0f MB/s | %8.0f MB/s\n",
               r.mb,
               throughput_mbs(r.mb, r.t_hint_populate),
               throughput_mbs(r.mb, r.t_no_prefault),
               throughput_mbs(r.mb, r.t_ifstream));
    }

    // Cleanup
    for (const auto& tf : files) {
        std::filesystem::remove(tf.path);
    }
}

TEST_F(MmapBenchmark, DISABLED_partial_prefault_offset_table) {
    constexpr size_t file_size_mb = 1200;
    constexpr size_t file_size = file_size_mb * 1024 * 1024;
    constexpr int warmup = 1;
    constexpr int runs = 5;

    auto file_path = generate_test_file(file_size);
    evict_cache(file_path, file_size);  // Flush dirty pages from file generation

    const std::vector<size_t> offsets_mb = {0, 1, 17, 500, 700, 800};
    const std::vector<size_t> region_sizes_mb = {10, 100, 500};

    // Pre-compute all results[size_idx][offset_idx]; -1 means "exceeds"
    std::vector<std::vector<long long>> results(region_sizes_mb.size(),
                                                std::vector<long long>(offsets_mb.size(), -1));
    for (size_t si = 0; si < region_sizes_mb.size(); ++si) {
        for (size_t oi = 0; oi < offsets_mb.size(); ++oi) {
            const size_t off_bytes = offsets_mb[oi] * 1024 * 1024;
            const size_t sz_bytes = region_sizes_mb[si] * 1024 * 1024;
            if (off_bytes + sz_bytes > file_size)
                continue;
            results[si][oi] = bench(
                [&]() {
                    strategy_hint_populate_partial(file_path, file_size, off_bytes, sz_bytes);
                },
                file_path,
                file_size,
                warmup,
                runs);
        }
    }

    // Print: sizes as rows, offsets as columns
    printf("\n--- partial prefault: hint_populate with offset ---\n");
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

    std::filesystem::remove(file_path);
}

}  // namespace ov::test
