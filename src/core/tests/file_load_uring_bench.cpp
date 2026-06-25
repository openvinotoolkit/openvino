// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <unistd.h>

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
#include <thread>
#include <tuple>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/io.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

#ifdef __linux__
#    include <fcntl.h>
#    include <sys/mman.h>
#endif

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov::test {
namespace {
const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));

struct TestFile {
    // size_t size_mb;
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

double throughput_mbs(size_t size_mib, long long ms) {
    if (ms <= 0)
        return 0.0;
    return static_cast<double>(size_mib) * 1000.0 / static_cast<double>(ms);
}

struct BenchScenario {
    std::function<void()> func;
    std::filesystem::path path;
    size_t size;
    int warmup_runs;
    int measured_runs;
};

struct BenchMeasurement {
    std::string name;
    BenchScenario scenario;
    long long time_ms;
};

[[maybe_unused]] long long bench(const BenchScenario& params) {
    return bench(params.func, params.path, params.size, params.warmup_runs, params.measured_runs);
}

namespace strategy {

// mlock forces every page resident before returning; munlock releases the pin without evicting.
// Bounded by RLIMIT_MEMLOCK -- no limit on a privileged process.
[[maybe_unused]] void mlock_munlock(const std::shared_ptr<ov::MappedMemory>& mapped) {
    ASSERT_EQ(::mlock(mapped->data(), mapped->size()), 0)
    /*
<< "mlock failed (errno=" << errno << "); check RLIMIT_MEMLOCK";
::munlock(mapped->data(), mapped->size()) */
    ;
}

// Note: the mmap destructor (munmap + close) runs inside the timed window;
void mmap_parallel_touch(const std::filesystem::path& path, size_t /*file_size*/) {
    auto mapped = load_mmap_object(path);
    util::vm_prefetch(mapped->data(), mapped->size(), std::thread::hardware_concurrency());
    // mlock_munlock(mapped);
}

void mmap_loop_touch(const std::filesystem::path& path, size_t /*file_size*/) {
    auto mapped = load_mmap_object(path);
    volatile uint8_t sink = 0;
    for (auto first = mapped->data(), last = first + mapped->size(); first < last; first += page_size) {
        sink += *first;
    }
    // mlock_munlock(mapped);
}

// Strategy: ifstream into a single pre-allocated buffer — one kernel→user copy
[[maybe_unused]] void ifstream_read(const std::filesystem::path& path, size_t file_size) {
    std::vector<char> read_buffer(file_size);
    std::ifstream f(path, std::ios::binary);
    f.read(read_buffer.data(), static_cast<std::streamsize>(file_size));
    volatile char sink = read_buffer[0] + read_buffer[file_size / 2] + read_buffer[file_size - 1];
    (void)sink;
}

void io_uring(const std::filesystem::path& path,
              size_t file_size,
              bool use_madvise_populate,
              //    bool use_sequential,
              bool use_fallback) {
    if (use_madvise_populate) {
        setenv("OV_UTIL_IO_URING_MADV_POPULATE", "1", 1);
    } else {
        unsetenv("OV_UTIL_IO_URING_MADV_POPULATE");
    }
    if (use_fallback) {
        setenv("OV_UTIL_IO_URING_FALLBACK", "1", 1);
    } else {
        unsetenv("OV_UTIL_IO_URING_FALLBACK");
    }

    auto mapped = load_mmap_object(path);
    util::io_populate_mmap(mapped->data(), mapped->size(), 0, 8);
    if (!use_madvise_populate)
        mlock_munlock(mapped);

    unsetenv("OV_UTIL_IO_URING_MADV_POPULATE");
    unsetenv("OV_UTIL_IO_URING_FALLBACK");
    // unsetenv("OV_UTIL_IO_URING_ACCESS_ADVISE");
}

}  // namespace strategy
}  // namespace

TEST(IOBenchmark, io_uring) {
#if 1
    constexpr int warmup = 0;
    constexpr int runs = 3;
    std::vector<size_t> sizes = {10, 100, 500, 1000};
    std::vector<int> advises = {MADV_NORMAL, MADV_SEQUENTIAL, MADV_RANDOM};
#else
    constexpr int warmup = 0;
    constexpr int runs = 1;
    std::vector<size_t> sizes = {10 /*, 100, 500, 1000 */};
    std::vector<int> advises = {MADV_NORMAL /* , MADV_SEQUENTIAL, MADV_RANDOM */};
#endif

    if constexpr (false) {
        std::vector<BenchMeasurement> benches;

        for (const auto sz : sizes) {
            const auto size_bytes = sz * 0x100000u;
            const auto path = generate_test_file(TestFile{size_bytes, {}});
            evict_cache(path, size_bytes);
            benches.push_back(BenchMeasurement{"io_uring",
                                               BenchScenario{[=]() {
                                                                 strategy::io_uring(path, size_bytes, true, false);
                                                             },
                                                             path,
                                                             size_bytes,
                                                             warmup,
                                                             runs},
                                               0});
        }
    } else {
        std::vector<TestFile> files;
        for (const auto sz : sizes) {
            TestFile tf{sz * 0x100000u, {}};
            tf.path = generate_test_file(tf);
            evict_cache(tf.path, tf.size_bytes);
            files.push_back(std::move(tf));
        }

        for (const auto& adv : advises) {
            printf("\n--- io_uring with access_advise %s ---\n",
                   adv == MADV_SEQUENTIAL ? "SEQUENTIAL" : (adv == MADV_RANDOM ? "RANDOM" : "NORMAL"));
            switch (adv) {
            case MADV_SEQUENTIAL:
                setenv("OV_UTIL_IO_URING_ACCESS_ADVISE", "SEQUENTIAL", 1);
                break;
            case MADV_RANDOM:
                setenv("OV_UTIL_IO_URING_ACCESS_ADVISE", "RANDOM", 1);
                break;
            default:
                unsetenv("OV_UTIL_IO_URING_ACCESS_ADVISE");
                break;
            }

#define BENCH(pop, fall)                                           \
    bench(                                                         \
        [&]() {                                                    \
            strategy::io_uring(tf.path, tf.size_bytes, pop, fall); \
        },                                                         \
        tf.path,                                                   \
        tf.size_bytes,                                             \
        warmup,                                                    \
        runs)

            struct Row {
                size_t mib;
                long long t_parallel_touch;
                long long t_loop_touch;
                long long t_hint_prefetch_uring_1;
                long long t_hint_prefetch_uring_2;
                long long t_hint_prefetch_uring_3;
                long long t_hint_prefetch_uring_4;
            };
            std::vector<Row> results;
            for (const auto& tf : files) {
                Row r{};
                r.mib = tf.size_bytes / 0x100000u;
                r.t_loop_touch = bench(
                    [&]() {
                        strategy::mmap_loop_touch(tf.path, tf.size_bytes);
                    },
                    tf.path,
                    tf.size_bytes,
                    warmup,
                    runs);
                r.t_parallel_touch = bench(
                    [&]() {
                        strategy::mmap_parallel_touch(tf.path, tf.size_bytes);
                    },
                    tf.path,
                    tf.size_bytes,
                    warmup,
                    runs);

                constexpr bool populate = true;
                constexpr bool fallback = true;
                r.t_hint_prefetch_uring_1 = BENCH(populate, !fallback);
                r.t_hint_prefetch_uring_2 = BENCH(!populate, !fallback);
                r.t_hint_prefetch_uring_3 = BENCH(populate, fallback);
                r.t_hint_prefetch_uring_4 = BENCH(!populate, fallback);
                results.push_back(r);
            }
#undef BENCH

            /*             printf("\n--- Latency (ms, mean of %d runs, cold cache) ---\n", runs);
                        printf("%-10s | %10s | %10s | %10s | %10s | %10s | %10s\n",
                               "Size (MB)",
                               "parallel touch",
                               "loop touch",
                               "uring 1",
                               "uring 2",
                               "uring 3",
                               "uring 4");
                        printf("%-10s-|-%10s-|-%10s-|-%10s-|-%10s-|-%10s-|-%10s\n",
                               "----------",
                               "----------",
                               "----------",
                               "----------",
                               "----------",
                               "----------",
                               "----------");
                        for (const auto& r : results) {
                            printf("%-10zu | %10lld | %10lld | %10lld | %10lld | %10lld | %10lld\n",
                                   r.mib,
                                   r.t_parallel_touch,
                                   r.t_loop_touch,
                                   r.t_hint_prefetch_uring_1,
                                   r.t_hint_prefetch_uring_2,
                                   r.t_hint_prefetch_uring_3,
                                   r.t_hint_prefetch_uring_4);
                        } */

            printf("\n--- Throughput (MB/s) ---\n");

            // std::vector<std::string> headers =
            //     {"Size (MB)", "parallel touch", "loop touch", "uring 1", "uring 2", "uring 3", "uring 4"};
            // std::string title_row;
            // std::string lines_row;
            // for (const auto& h : headers) {
            // }

            printf("%-10s | %14s | %10s | %7s | %7s | %7s | %7s\n",
                   "Size (MB)",
                   "parallel touch",
                   "loop touch",
                   "uring 1",
                   "uring 2",
                   "uring 3",
                   "uring 4");
            printf("%-10s-|-%14s-|-%10s-|-%7s-|-%7s-|-%7s-|-%7s\n",
                   "----------",
                   "--------------",
                   "----------",
                   "-------",
                   "-------",
                   "-------",
                   "-------");
            for (const auto& r : results) {
                printf("%-10zu | %14.0f | %10.0f | %7.0f | %7.0f | %7.0f | %7.0f\n",
                       r.mib,
                       throughput_mbs(r.mib, r.t_parallel_touch),
                       throughput_mbs(r.mib, r.t_loop_touch),
                       throughput_mbs(r.mib, r.t_hint_prefetch_uring_1),
                       throughput_mbs(r.mib, r.t_hint_prefetch_uring_2),
                       throughput_mbs(r.mib, r.t_hint_prefetch_uring_3),
                       throughput_mbs(r.mib, r.t_hint_prefetch_uring_4));
            }
        }
            printf("\nuring 1: io_populate_mmap with `io_uring_prep_madvise(..., MADV_POPULATE_READ)'\n");
            printf("uring 2: io_populate_mmap with `io_uring_prep_madvise(..., MADV_WILLNEED)'\n");
            printf("uring 3: io_populate_mmap falling back to `madvise(..., MADV_POPULATE_READ)\n");
            printf("uring 4: io_populate_mmap falling back to `madvise(..., MADV_WILLNEED)\n\n");
    }
}
}  // namespace ov::test
