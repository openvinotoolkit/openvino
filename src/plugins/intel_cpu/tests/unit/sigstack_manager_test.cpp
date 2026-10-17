// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__linux__)

#    include "sigstack_manager.h"

#    include <gtest/gtest.h>
#    include <sys/mman.h>
#    include <unistd.h>

#    include <csignal>
#    include <cstdint>
#    include <memory>

using namespace ov::intel_cpu;

namespace {

stack_t current_altstack() {
    stack_t stack{};
    EXPECT_EQ(sigaltstack(nullptr, &stack), 0);
    return stack;
}

bool is_mapped(const void* address) {
    const auto page_size = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    auto* page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) & ~(page_size - 1));
    // msync fails with ENOMEM if the range is not mapped
    return msync(page, page_size, MS_ASYNC) == 0;
}

class SigAltStackSetupTest : public ::testing::Test {
protected:
    void SetUp() override {
        initial = current_altstack();
    }

    void TearDown() override {
        sigaltstack(&initial, nullptr);
    }

    stack_t initial{};
};

TEST_F(SigAltStackSetupTest, RestoresInitialStackWhenDestroyedInOrder) {
    auto first = std::make_unique<SigAltStackSetup>();
    auto second = std::make_unique<SigAltStackSetup>();
    second.reset();
    first.reset();

    const auto current = current_altstack();
    EXPECT_EQ(current.ss_sp, initial.ss_sp);
    EXPECT_EQ(current.ss_flags & SS_DISABLE, initial.ss_flags & SS_DISABLE);
}

TEST_F(SigAltStackSetupTest, KeepsCurrentStackMappedWhenDestroyedOutOfOrder) {
    auto first = std::make_unique<SigAltStackSetup>();
    auto second = std::make_unique<SigAltStackSetup>();
    first.reset();
    second.reset();

    const auto current = current_altstack();
    if (!(current.ss_flags & SS_DISABLE) && current.ss_sp != initial.ss_sp) {
        EXPECT_TRUE(is_mapped(current.ss_sp));
        EXPECT_TRUE(is_mapped(static_cast<const char*>(current.ss_sp) + current.ss_size - 1));
    }
}

TEST_F(SigAltStackSetupTest, KeepsCurrentStackMappedWhenOuterInstanceIsDestroyedFirst) {
    auto first = std::make_unique<SigAltStackSetup>();
    auto second = std::make_unique<SigAltStackSetup>();
    const auto second_stack = current_altstack();
    first.reset();

    const auto current = current_altstack();
    EXPECT_EQ(current.ss_sp, second_stack.ss_sp);
    EXPECT_TRUE(is_mapped(current.ss_sp));
}

}  // namespace

#endif  // __linux__
