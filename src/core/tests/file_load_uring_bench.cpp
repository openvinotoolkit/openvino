// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined ENABLE_IO_URING

#    include <gtest/gtest.h>
#    include <unistd.h>

#    include <algorithm>
#    include <cerrno>
#    include <chrono>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <filesystem>
#    include <fstream>
#    include <functional>
#    include <iostream>
#    include <numeric>
#    include <string>
#    include <thread>
#    include <tuple>
#    include <vector>

#    include "openvino/util/file_util.hpp"
#    include "openvino/util/io.hpp"
#    include "openvino/util/memory.hpp"
#    include "openvino/util/mmap_object.hpp"

#    ifdef __linux__
#        include <fcntl.h>
#        include <sys/mman.h>
#    endif

#    include "common_test_utils/common_utils.hpp"
#    include "common_test_utils/file_utils.hpp"

namespace ov::test {
namespace {
const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));

struct TestFile {
    size_t size_bytes;
    std::filesystem::path path;
};

std::filesystem::path generate_test_file(const TestFile& tf) {
    const auto dir = getenv("OV_UTIL_IO_BENCH_DIR");
    auto path = dir ? std::filesystem::path{dir} : std::filesystem::current_path();
    path /= "test_file" + std::to_string(tf.size_bytes / 0x100000u) + "mib.bin";

    if (util::file_exists(path) && std::filesystem::file_size(path) == tf.size_bytes) {
        std::cout << "Test file already exists: " << path << ", skipping generation." << std::endl;
        return path;
    }
    std::vector<uint8_t> chunk(1024 * 1024);
    for (size_t i = 0; i < chunk.size(); ++i)
        chunk[i] = static_cast<uint8_t>(i % 251);
    std::ofstream f(path, std::ios::binary);
    for (size_t written = 0; written < tf.size_bytes; written += chunk.size()) {
        const auto to_write = std::min(chunk.size(), tf.size_bytes - written);
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

    static bool warned = false;
    if (!warned) {
        std::cout << "[WARNING] No access to /proc/sys/vm/drop_caches, falling back to "
                     "posix_fadvise(DONTNEED). Results may be unreliable (kernel can ignore the hint)."
                  << std::endl;
        warned = true;
    }

    // Fallback: best-effort fadvise.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, static_cast<off_t>(file_size), POSIX_FADV_DONTNEED);
        ::close(fd);
    }
}

long long bench(const std::function<void(ov::MappedMemory&)>& fn,
                const std::filesystem::path& path,
                size_t file_size,
                int warmup_runs = 1,
                int measured_runs = 5) {
    for (int i = 0; i < warmup_runs; ++i) {
        evict_cache(path, file_size);
        auto mapped = load_mmap_object(path);
        fn(*mapped);
    }
    long long total = 0;
    for (int i = 0; i < measured_runs; ++i) {
        evict_cache(path, file_size);
        auto mapped = load_mmap_object(path);
        total += measure_ms([&]() {
            fn(*mapped);
        });
    }
    return total / measured_runs;
}

double throughput_mbs(size_t size_mib, long long ms) {
    if (ms <= 0)
        return 0.0;
    return static_cast<double>(size_mib) * 1000.0 / static_cast<double>(ms);
}

void page_touch(ov::MappedMemory& mapped, size_t num_threads) {
    util::vm_prefetch(mapped.data(), mapped.size(), num_threads);
}

void io_uring(ov::MappedMemory& mapped, size_t depth) {
    util::io_populate_mmap(mapped.data(), mapped.size(), 0, depth);
}
}  // namespace

TEST(FileLoadBench, io_uring_vs_pg_touch) {
    constexpr int warmup = 0;
    constexpr int runs = 3;
    std::vector<size_t> sizes = {10, 100, 500, 1000};

    std::vector<TestFile> files;
    for (const auto sz : sizes) {
        TestFile tf{sz * 0x100000u, {}};
        tf.path = generate_test_file(tf);
        evict_cache(tf.path, tf.size_bytes);
        files.push_back(std::move(tf));
    }

    using BenchFn = void (*)(ov::MappedMemory&, size_t);
    const std::vector<std::tuple<BenchFn, size_t, std::string>> bench_configs = {
        {page_touch, 12, "pg touch"},
        {page_touch, 24, "pg touch"},
        {io_uring, 8, "io uring"},
        {io_uring, 32, "io uring"},
        {io_uring, 128, "io uring"},
        {io_uring, 256, "io uring"},
        {io_uring, 1024, "io uring"},
        {io_uring, 4096, "io uring"},
    };

    struct Row {
        size_t mib;
        std::vector<long long> timings;
    };
    std::vector<Row> results;
    for (const auto& tf : files) {
        Row r{};
        r.mib = tf.size_bytes / 0x100000u;
        for (const auto& [fn, dep, label] : bench_configs) {
            r.timings.push_back(bench(
                [&](ov::MappedMemory& mapped) {
                    fn(mapped, dep);
                },
                tf.path,
                tf.size_bytes,
                warmup,
                runs));
        }
        results.push_back(std::move(r));
    }

    std::vector<std::pair<std::string, int>> col_headers;
    auto make_col = [](std::string s) {
        return std::pair<std::string, int>{s, static_cast<int>(s.size())};
    };
    col_headers.push_back(make_col("Size (MB)"));
    for (const auto& cfg : bench_configs)
        col_headers.push_back(make_col(std::get<2>(cfg) + " " + std::to_string(std::get<1>(cfg))));

    auto print_header = [&]() {
        printf("%-*s", col_headers[0].second, col_headers[0].first.c_str());
        for (size_t i = 1; i < col_headers.size(); ++i)
            printf(" | %*s", col_headers[i].second, col_headers[i].first.c_str());
        printf("\n");
    };

    auto print_separator = [&]() {
        printf("%-*s", col_headers[0].second, std::string(col_headers[0].second, '-').c_str());
        for (size_t i = 1; i < col_headers.size(); ++i)
            printf("-|-%*s", col_headers[i].second, std::string(col_headers[i].second, '-').c_str());
        printf("\n");
    };

    printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
    print_header();
    print_separator();
    for (const auto& r : results) {
        printf("%-*zu", col_headers[0].second, r.mib);
        for (size_t i = 0; i < r.timings.size(); ++i)
            printf(" | %*lld", col_headers[i + 1].second, r.timings[i]);
        printf("\n");
    }

    printf("\n--- Throughput (MB/s) ---\n");
    print_header();
    print_separator();
    for (const auto& r : results) {
        printf("%-*zu", col_headers[0].second, r.mib);
        for (size_t i = 0; i < r.timings.size(); ++i)
            printf(" | %*.0f", col_headers[i + 1].second, throughput_mbs(r.mib, r.timings[i]));
        printf("\n");
    }
    printf("\n");
}
}  // namespace ov::test

#endif
