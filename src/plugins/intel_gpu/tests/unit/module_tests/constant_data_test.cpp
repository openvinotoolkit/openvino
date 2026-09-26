// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/constant_data.hpp"

#include <gtest/gtest.h>

#include <future>
#include <vector>

namespace {
class copy_executor : public ov::threading::ITaskExecutor {
public:
    void run(ov::threading::Task task) override {
        ++submitted;
        workers.push_back(std::async(std::launch::async, std::move(task)));
    }

    size_t submitted = 0;
    std::vector<std::future<void>> workers;
};

void check_copy(size_t size, size_t max_threads, bool use_executor, size_t expected_tasks) {
    constexpr size_t guard_size = 7;
    std::vector<unsigned char> source(size + 2 * guard_size);
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<unsigned char>(i * 37 + 11);
    std::vector<unsigned char> destination(source.size(), 0xa5);
    copy_executor executor;

    ov::intel_gpu::copy_constant_data(destination.data() + guard_size, source.data() + guard_size, size, use_executor ? &executor : nullptr, max_threads);

    EXPECT_EQ(executor.submitted, expected_tasks);
    EXPECT_TRUE(std::equal(source.begin() + guard_size, source.end() - guard_size, destination.begin() + guard_size));
    for (size_t i = 0; i < guard_size; ++i) {
        EXPECT_EQ(destination[i], 0xa5);
        EXPECT_EQ(destination[destination.size() - 1 - i], 0xa5);
    }
}
}  // namespace

TEST(constant_data, empty_copy_does_not_submit_tasks) {
    copy_executor executor;
    EXPECT_NO_THROW(ov::intel_gpu::copy_constant_data(nullptr, nullptr, 0, &executor, 4));
    EXPECT_EQ(executor.submitted, 0);
}

TEST(constant_data, small_and_threshold_copies_use_caller) {
    check_copy(19, 8, true, 0);
    check_copy(8 * 1024 * 1024 - 1, 8, true, 0);
}

TEST(constant_data, parallel_copy_preserves_unaligned_bytes_and_remainder) {
    check_copy(8 * 1024 * 1024, 8, true, 2);
    check_copy(17 * 1024 * 1024 + 19, 3, true, 3);
    check_copy(17 * 1024 * 1024 + 19, 8, true, 4);
}

TEST(constant_data, missing_executor_and_single_worker_use_caller) {
    check_copy(9 * 1024 * 1024 + 3, 8, false, 0);
    check_copy(9 * 1024 * 1024 + 3, 1, true, 0);
    check_copy(9 * 1024 * 1024 + 3, 0, true, 0);
}
