// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sigstack_manager.h"

#include <cstring>

#if defined(__linux__)
#    include <sys/auxv.h>
#    include <sys/mman.h>

#    include <csignal>
#endif

namespace ov::intel_cpu {

#if defined(__linux__)

#    ifndef AT_MINSIGSTKSZ
#        define AT_MINSIGSTKSZ 51
#    endif

SigAltStackSetup::SigAltStackSetup() {
    memset(&old_stack, 0, sizeof(old_stack));
    memset(&new_stack, 0, sizeof(new_stack));

    auto minsigstksz = getauxval(AT_MINSIGSTKSZ);
    auto new_size = minsigstksz + SIGSTKSZ;  // NOLINT(misc-include-cleaner) bug in clang-tidy
    void* altstack = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (altstack == MAP_FAILED) {
        return;
    }
    new_stack.ss_size = new_size;
    new_stack.ss_sp = altstack;
    auto rc = sigaltstack(&new_stack, &old_stack);  // NOLINT(misc-include-cleaner) bug in clang-tidy
    if (rc) {
        munmap(new_stack.ss_sp, new_stack.ss_size);
        new_stack.ss_sp = nullptr;
        new_stack.ss_size = 0;
        return;
    }
}

SigAltStackSetup::~SigAltStackSetup() {
    stack_t current_stack;
    if (new_stack.ss_sp) {
        // restore old stack if new_stack is still the current one
        if (sigaltstack(nullptr, &current_stack) == 0) {
            if (current_stack.ss_sp == new_stack.ss_sp) {
                sigaltstack(&old_stack, nullptr);
            }
        }
        munmap(new_stack.ss_sp, new_stack.ss_size);
        new_stack.ss_sp = nullptr;
        new_stack.ss_size = 0;
    }
}

#endif  // __linux__

}  // namespace ov::intel_cpu
