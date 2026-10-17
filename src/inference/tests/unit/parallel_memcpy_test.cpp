// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/parallel_memcpy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <future>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace {
class CopyExecutor : public ov::threading::ITaskExecutor {
public:
    void run(ov::threading::Task task) override {
        ++submitted;
        workers.push_back(std::async(std::launch::async, std::move(task)));
    }

    size_t submitted = 0;
    std::vector<std::future<void>> workers;
};

template <class Copy>
void check_copy(size_t size, const Copy& copy) {
    constexpr size_t guard = 7;
    std::vector<unsigned char> source(size + 2 * guard);
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<unsigned char>(i * 37 + 11);
    std::vector<unsigned char> destination(source.size(), 0xa5);
    copy(destination.data() + guard, source.data() + guard, size);
    EXPECT_TRUE(std::equal(source.begin() + guard, source.end() - guard, destination.begin() + guard));
    for (size_t i = 0; i < guard; ++i) {
        EXPECT_EQ(destination[i], 0xa5);
        EXPECT_EQ(destination[destination.size() - 1 - i], 0xa5);
    }
}
}  // namespace

TEST(ParallelMemcpy, empty_buffers) {
    CopyExecutor executor;
    ov::threading::parallel_memcpy(nullptr, nullptr, 0);
    ov::threading::parallel_memcpy(nullptr, nullptr, 0, executor, 4);
    EXPECT_EQ(executor.submitted, 0);
}

TEST(ParallelMemcpy, current_arena_preserves_unaligned_bytes_and_remainder) {
    for (size_t size : {1UL, 3UL, 19UL, 1024UL * 1024 + 19}) {
        check_copy(size, [](void* dst, const void* src, size_t count) {
            ov::threading::parallel_memcpy(dst, src, count);
        });
    }
}

TEST(ParallelMemcpy, executor_completes_copies_before_returning) {
    CopyExecutor executor;
    check_copy(17 * 1024 * 1024 + 19, [&](void* dst, const void* src, size_t count) {
        ov::threading::parallel_memcpy(dst, src, count, executor, 3);
    });
    EXPECT_EQ(executor.submitted, 3);
}

TEST(ParallelMemcpy, executor_does_not_submit_empty_parts) {
    CopyExecutor executor;
    check_copy(3, [&](void* dst, const void* src, size_t count) {
        ov::threading::parallel_memcpy(dst, src, count, executor, 8);
    });
    EXPECT_EQ(executor.submitted, 3);
}

TEST(ParallelMemcpy, zero_or_one_task_uses_caller) {
    CopyExecutor executor;
    for (size_t tasks : {0UL, 1UL}) {
        check_copy(1024 * 1024 + 19, [&](void* dst, const void* src, size_t count) {
            ov::threading::parallel_memcpy(dst, src, count, executor, tasks);
        });
    }
    EXPECT_EQ(executor.submitted, 0);
}

#if OV_THREAD_USE_TBB
TEST(ParallelMemcpy, works_in_restricted_and_nested_arenas) {
    tbb::task_arena arena(2);
    arena.execute([&] {
        ov::parallel_nt(2, [&](size_t, size_t) {
            check_copy(1024 * 1024 + 19, [](void* dst, const void* src, size_t count) {
                ov::threading::parallel_memcpy(dst, src, count);
            });
        });
        tbb::task_arena serial_arena(1);
        serial_arena.execute([] {
            check_copy(1024 * 1024 + 19, [](void* dst, const void* src, size_t count) {
                ov::threading::parallel_memcpy(dst, src, count);
            });
        });
    });
}
#endif
