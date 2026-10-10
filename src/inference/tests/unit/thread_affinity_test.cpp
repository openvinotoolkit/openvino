// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dev/threading/thread_affinity.hpp"

#include <gtest/gtest.h>

#include <bitset>

using namespace testing;
using namespace ov;

namespace {

#if defined(__linux__)

// RAII guard restoring the calling thread's affinity mask on destruction.
class ScopedThreadMask {
public:
    ScopedThreadMask() : _valid(false) {
        ov::threading::CpuSet mask;
        int ncpus = 0;
        std::tie(mask, ncpus) = ov::threading::query_thread_mask();
        if (nullptr != mask) {
            _ncpus = ncpus;
            _mask = std::move(mask);
            _valid = true;
        }
    }
    ~ScopedThreadMask() {
        if (_valid) {
            ov::threading::pin_current_thread_by_mask(_ncpus, _mask);
        }
    }
    ScopedThreadMask(const ScopedThreadMask&) = delete;
    ScopedThreadMask& operator=(const ScopedThreadMask&) = delete;

private:
    ov::threading::CpuSet _mask;
    int _ncpus = 0;
    bool _valid;
};

// Regression test for the process-affinity baseline cache: get_process_mask() must return the
// unpolluted process baseline even if the calling thread's affinity is temporarily restricted
// (Linux maps getpid() to the main thread TID, so a plain sched_getaffinity(getpid(),...) from a
// pinned thread would return the thread's narrow mask).
TEST(ThreadAffinity, ProcessMaskStaysAtBaselineWhileThreadIsPinned) {
    auto baseline = ov::threading::get_process_mask();
    auto& baseline_mask = std::get<0>(baseline);
    const int baseline_ncpus = std::get<1>(baseline);
    ASSERT_NE(baseline_mask, nullptr);
    ASSERT_GT(baseline_ncpus, 0);
    const size_t size = CPU_ALLOC_SIZE(baseline_ncpus);

    // Requires at least two CPUs to distinguish the narrowed thread mask from the baseline.
    if (CPU_COUNT_S(size, baseline_mask.get()) < 2) {
        GTEST_SKIP() << "Process has fewer than 2 CPUs available; test not applicable.";
    }

    ScopedThreadMask restore;

    // Pick a single CPU to restrict the current thread to.
    int target_cpu = -1;
    for (int i = 0; i < baseline_ncpus; i++) {
        if (CPU_ISSET_S(i, size, baseline_mask.get())) {
            target_cpu = i;
            break;
        }
    }
    ASSERT_GE(target_cpu, 0);

    ov::threading::CpuSet narrow{CPU_ALLOC(baseline_ncpus)};
    ASSERT_NE(narrow, nullptr);
    CPU_ZERO_S(size, narrow.get());
    CPU_SET_S(target_cpu, size, narrow.get());

    // Pin the calling thread to a single CPU.
    ASSERT_TRUE(ov::threading::pin_current_thread_by_mask(baseline_ncpus, narrow));

    // query_thread_mask() must now reflect the restricted (narrow) mask.
    auto thread_mask = ov::threading::query_thread_mask();
    auto& thread_mask_ptr = std::get<0>(thread_mask);
    ASSERT_NE(thread_mask_ptr, nullptr);
    ASSERT_EQ(CPU_COUNT_S(size, thread_mask_ptr.get()), 1);
    ASSERT_TRUE(CPU_ISSET_S(target_cpu, size, thread_mask_ptr.get()));

    // get_process_mask() must still return the full baseline, not the narrowed thread mask.
    auto baseline_after = ov::threading::get_process_mask();
    auto& baseline_after_mask = std::get<0>(baseline_after);
    ASSERT_NE(baseline_after_mask, nullptr);
    ASSERT_TRUE(CPU_EQUAL_S(size, baseline_after_mask.get(), baseline_mask.get()));
    // The baseline must contain the restricted CPU as well (it is a superset).
    ASSERT_TRUE(CPU_ISSET_S(target_cpu, size, baseline_after_mask.get()));
}

#elif defined(_WIN32)
#    include <windows.h>

// RAII guard restoring the calling thread's affinity mask on destruction.
class ScopedThreadMask {
public:
    ScopedThreadMask() : _valid(false) {
        ov::threading::CpuSet mask;
        int ncpus = 0;
        std::tie(mask, ncpus) = ov::threading::query_thread_mask();
        if (nullptr != mask) {
            _mask = std::move(mask);
            _valid = true;
        }
    }
    ~ScopedThreadMask() {
        if (_valid) {
            ov::threading::pin_current_thread_by_mask(0, _mask);
        }
    }
    ScopedThreadMask(const ScopedThreadMask&) = delete;
    ScopedThreadMask& operator=(const ScopedThreadMask&) = delete;

private:
    ov::threading::CpuSet _mask;
    bool _valid;
};

// The calling thread's affinity must be restored (see ScopedThreadMask), and get_process_mask()
// must keep returning the unpolluted process baseline even while the thread is temporarily pinned.
TEST(ThreadAffinity, ProcessMaskStaysAtBaselineWhileThreadIsPinned) {
    auto baseline = ov::threading::get_process_mask();
    auto& baseline_mask = std::get<0>(baseline);
    ASSERT_NE(baseline_mask, nullptr);

    // Requires at least two CPUs to distinguish the narrowed thread mask from the baseline.
    if (std::bitset<sizeof(DWORD_PTR) * 8>(*baseline_mask).count() < 2) {
        GTEST_SKIP() << "Process has fewer than 2 CPUs available; test not applicable.";
    }

    ScopedThreadMask restore;

    // Pick a single CPU to restrict the current thread to.
    int target_cpu = -1;
    for (int i = 0; i < static_cast<int>(sizeof(DWORD_PTR) * 8); i++) {
        if ((*baseline_mask & (DWORD_PTR(1) << i)) != 0) {
            target_cpu = i;
            break;
        }
    }
    ASSERT_GE(target_cpu, 0);

    // Pin the calling thread to a single CPU.
    ov::threading::CpuSet narrow = std::make_unique<ov::threading::cpu_set_t>(DWORD_PTR(1) << target_cpu);
    ASSERT_NE(narrow, nullptr);
    ASSERT_TRUE(ov::threading::pin_current_thread_by_mask(0, narrow));

    // query_thread_mask() must now reflect the restricted (narrow) mask.
    auto thread_mask = ov::threading::query_thread_mask();
    auto& thread_mask_ptr = std::get<0>(thread_mask);
    ASSERT_NE(thread_mask_ptr, nullptr);
    ASSERT_EQ(*thread_mask_ptr, DWORD_PTR(1) << target_cpu);

    // get_process_mask() must still return the full baseline, not the narrowed thread mask.
    auto baseline_after = ov::threading::get_process_mask();
    auto& baseline_after_mask = std::get<0>(baseline_after);
    ASSERT_NE(baseline_after_mask, nullptr);
    ASSERT_EQ(*baseline_after_mask, *baseline_mask);
    // The baseline must contain the restricted CPU as well (it is a superset).
    ASSERT_TRUE((*baseline_after_mask & (DWORD_PTR(1) << target_cpu)) != 0);
}

#endif  // defined(__linux__) || defined(_WIN32)

}  // namespace