// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dev/threading/thread_affinity.hpp"

#include <gtest/gtest.h>

#include <bitset>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#endif

namespace {

#if defined(__linux__) || defined(_WIN32)

// Regression test for the process-affinity baseline cache: get_process_mask() must return the
// unpolluted process baseline even if the calling thread's affinity is temporarily restricted
// (Linux maps getpid() to the main thread TID, so a plain sched_getaffinity(getpid(),...) from a
// pinned thread would return the thread's narrow mask).
TEST(ThreadAffinity, ProcessMaskStaysAtBaselineWhileThreadIsPinned) {
    auto baseline = ov::threading::get_process_mask();
    auto& baseline_mask = std::get<0>(baseline);
    const int baseline_ncpus = std::get<1>(baseline);
    ASSERT_NE(baseline_mask, nullptr);
#    if defined(__linux__)
    ASSERT_GT(baseline_ncpus, 0);
    const size_t size = CPU_ALLOC_SIZE(baseline_ncpus);

    // Requires at least two CPUs to distinguish the narrowed thread mask from the baseline.
    if (CPU_COUNT_S(size, baseline_mask.get()) < 2) {
        GTEST_SKIP() << "Process has fewer than 2 CPUs available; test not applicable.";
    }

    ov::threading::ScopedThreadAffinity restore;

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
#    elif defined(_WIN32)
    // Requires at least two CPUs to distinguish the narrowed thread mask from the baseline.
    if (std::bitset<sizeof(DWORD_PTR) * 8>(*baseline_mask).count() < 2) {
        GTEST_SKIP() << "Process has fewer than 2 CPUs available; test not applicable.";
    }

    ov::threading::ScopedThreadAffinity restore;

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
    ov::threading::CpuSet narrow =
        std::make_unique<ov::threading::cpu_set_t>(static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << target_cpu));
    ASSERT_NE(narrow, nullptr);
    ASSERT_TRUE(ov::threading::pin_current_thread_by_mask(0, narrow));

    // query_thread_mask() must now reflect the restricted (narrow) mask.
    auto thread_mask = ov::threading::query_thread_mask();
    auto& thread_mask_ptr = std::get<0>(thread_mask);
    ASSERT_NE(thread_mask_ptr, nullptr);
    ASSERT_EQ(static_cast<ov::threading::cpu_set_t>(*thread_mask_ptr),
              static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << target_cpu));

    // get_process_mask() must still return the full baseline, not the narrowed thread mask.
    auto baseline_after = ov::threading::get_process_mask();
    auto& baseline_after_mask = std::get<0>(baseline_after);
    ASSERT_NE(baseline_after_mask, nullptr);
    ASSERT_EQ(static_cast<ov::threading::cpu_set_t>(*baseline_after_mask),
              static_cast<ov::threading::cpu_set_t>(*baseline_mask));
    // The baseline must contain the restricted CPU as well (it is a superset).
    ASSERT_TRUE((*baseline_after_mask & (DWORD_PTR(1) << target_cpu)) != 0);
#    endif  // defined(__linux__) || defined(_WIN32)
}

// Verifies that if an exception is thrown in a block guarded by ScopedThreadAffinity, the calling
// thread's original affinity mask is accurately restored when the exception is caught.
TEST(ThreadAffinity, ScopedThreadAffinityRestoresOnException) {
#    if defined(__linux__)
    const int ncpus = std::get<1>(ov::threading::query_thread_mask());
    ASSERT_GT(ncpus, 0);
    const size_t size = CPU_ALLOC_SIZE(ncpus);
    const auto before = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(before), nullptr);

    try {
        ov::threading::ScopedThreadAffinity guard;

        // Restrict the thread to the first available CPU.
        int target_cpu = -1;
        for (int i = 0; i < ncpus; i++) {
            if (CPU_ISSET_S(i, size, std::get<0>(before).get())) {
                target_cpu = i;
                break;
            }
        }
        ASSERT_GE(target_cpu, 0);
        ov::threading::CpuSet narrow{CPU_ALLOC(ncpus)};
        CPU_ZERO_S(size, narrow.get());
        CPU_SET_S(target_cpu, size, narrow.get());
        ASSERT_TRUE(ov::threading::pin_current_thread_by_mask(ncpus, narrow));

        // Confirm the thread is now pinned to a single CPU, then throw.
        auto pinned = ov::threading::query_thread_mask();
        ASSERT_EQ(CPU_COUNT_S(size, std::get<0>(pinned).get()), 1);
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }

    // The guard's destructor must have restored the original mask during unwinding.
    auto after = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(after), nullptr);
    ASSERT_TRUE(CPU_EQUAL_S(size, std::get<0>(after).get(), std::get<0>(before).get()));
#    elif defined(_WIN32)
    const auto before = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(before), nullptr);

    try {
        ov::threading::ScopedThreadAffinity guard;

        // Restrict the thread to the first available CPU.
        int target_cpu = -1;
        for (int i = 0; i < static_cast<int>(sizeof(DWORD_PTR) * 8); i++) {
            if ((*std::get<0>(before) & (DWORD_PTR(1) << i)) != 0) {
                target_cpu = i;
                break;
            }
        }
        ASSERT_GE(target_cpu, 0);
        ov::threading::CpuSet narrow = std::make_unique<ov::threading::cpu_set_t>(
            static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << target_cpu));
        ASSERT_TRUE(ov::threading::pin_current_thread_by_mask(0, narrow));

        // Confirm the thread is now pinned to a single CPU, then throw.
        auto pinned = ov::threading::query_thread_mask();
        ASSERT_EQ(*std::get<0>(pinned), static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << target_cpu));
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }

    // The guard's destructor must have restored the original mask during unwinding.
    auto after = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(after), nullptr);
    ASSERT_EQ(static_cast<ov::threading::cpu_set_t>(*std::get<0>(after)),
              static_cast<ov::threading::cpu_set_t>(*std::get<0>(before)));
#    endif
}

// Verifies that nested ScopedThreadAffinity guards restore the calling thread's mask in the correct
// (reverse) order, so inner guards are restored before outer ones.
TEST(ThreadAffinity, ScopedThreadAffinityNestedUnwinding) {
#    if defined(__linux__)
    const int ncpus = std::get<1>(ov::threading::query_thread_mask());
    ASSERT_GT(ncpus, 0);
    const size_t size = CPU_ALLOC_SIZE(ncpus);
    const auto before = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(before), nullptr);
    // Requires at least two CPUs for two distinct single-CPU masks.
    if (CPU_COUNT_S(size, std::get<0>(before).get()) < 2) {
        GTEST_SKIP() << "Process has fewer than 2 CPUs available; test not applicable.";
    }

    auto set_mask_of_cpu = [&](int cpu) {
        ov::threading::CpuSet m{CPU_ALLOC(ncpus)};
        CPU_ZERO_S(size, m.get());
        CPU_SET_S(cpu, size, m.get());
        return ov::threading::pin_current_thread_by_mask(ncpus, m);
    };

    // Pick two distinct available CPUs.
    int cpu_a = -1;
    int cpu_b = -1;
    for (int i = 0; i < ncpus; i++) {
        if (!CPU_ISSET_S(i, size, std::get<0>(before).get()))
            continue;
        if (cpu_a < 0) {
            cpu_a = i;
        } else {
            cpu_b = i;
            break;
        }
    }
    ASSERT_NE(cpu_a, -1);
    ASSERT_NE(cpu_b, -1);

    try {
        ov::threading::ScopedThreadAffinity outer;
        ASSERT_TRUE(set_mask_of_cpu(cpu_a));
        {
            ov::threading::ScopedThreadAffinity inner;
            ASSERT_TRUE(set_mask_of_cpu(cpu_b));

            // Inner guard must have snapshotted the mask pinned in the outer scope (cpu_a).
            auto inner_snapshot = ov::threading::query_thread_mask();
            ASSERT_EQ(CPU_COUNT_S(size, std::get<0>(inner_snapshot).get()), 1);
            ASSERT_TRUE(CPU_ISSET_S(cpu_b, size, std::get<0>(inner_snapshot).get()));
        }

        // Close the inner scope normally: the inner guard restores the outer-scope mask (cpu_a).
        auto inner_restored = ov::threading::query_thread_mask();
        ASSERT_EQ(CPU_COUNT_S(size, std::get<0>(inner_restored).get()), 1);
        ASSERT_TRUE(CPU_ISSET_S(cpu_a, size, std::get<0>(inner_restored).get()));

        // Throw inside the outer scope so the outer guard unwinds and restores before-mask.
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }

    // After unwinding, the outer guard must have restored the original thread mask.
    auto after = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(after), nullptr);
    ASSERT_TRUE(CPU_EQUAL_S(size, std::get<0>(after).get(), std::get<0>(before).get()));
#    elif defined(_WIN32)
    const auto before = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(before), nullptr);
    // Requires at least two CPUs for two distinct single-CPU masks.
    if (std::bitset<sizeof(DWORD_PTR) * 8>(*std::get<0>(before)).count() < 2) {
        GTEST_SKIP() << "Process has fewer than 2 CPUs available; test not applicable.";
    }

    auto set_mask_of_cpu = [&](int cpu) {
        ov::threading::CpuSet m =
            std::make_unique<ov::threading::cpu_set_t>(static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << cpu));
        return ov::threading::pin_current_thread_by_mask(0, m);
    };

    // Pick two distinct available CPUs.
    int cpu_a = -1;
    int cpu_b = -1;
    for (int i = 0; i < static_cast<int>(sizeof(DWORD_PTR) * 8); i++) {
        if (!(*std::get<0>(before) & (DWORD_PTR(1) << i)))
            continue;
        if (cpu_a < 0) {
            cpu_a = i;
        } else {
            cpu_b = i;
            break;
        }
    }
    ASSERT_NE(cpu_a, -1);
    ASSERT_NE(cpu_b, -1);

    try {
        ov::threading::ScopedThreadAffinity outer;
        ASSERT_TRUE(set_mask_of_cpu(cpu_a));
        {
            ov::threading::ScopedThreadAffinity inner;
            ASSERT_TRUE(set_mask_of_cpu(cpu_b));

            // Inner guard must have snapshotted the mask pinned in the outer scope (cpu_a).
            auto inner_snapshot = ov::threading::query_thread_mask();
            ASSERT_EQ(static_cast<ov::threading::cpu_set_t>(*std::get<0>(inner_snapshot)),
                      static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << cpu_b));
        }

        // Close the inner scope normally: the inner guard restores the outer-scope mask (cpu_a).
        auto inner_restored = ov::threading::query_thread_mask();
        ASSERT_EQ(static_cast<ov::threading::cpu_set_t>(*std::get<0>(inner_restored)),
                  static_cast<ov::threading::cpu_set_t>(DWORD_PTR(1) << cpu_a));

        // Throw inside the outer scope so the outer guard unwinds and restores before-mask.
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }

    // After unwinding, the outer guard must have restored the original thread mask.
    auto after = ov::threading::query_thread_mask();
    ASSERT_NE(std::get<0>(after), nullptr);
    ASSERT_EQ(static_cast<ov::threading::cpu_set_t>(*std::get<0>(after)),
              static_cast<ov::threading::cpu_set_t>(*std::get<0>(before)));
#    endif
}

#endif  // defined(__linux__) || defined(_WIN32)

}  // namespace
