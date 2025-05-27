// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__linux__)
#    include <csignal>
#endif

namespace ov::intel_cpu {

#if defined(__linux__)
class SigAltStackSetup {
public:
    SigAltStackSetup();

    ~SigAltStackSetup();

private:
    stack_t new_stack{nullptr};  // NOLINT(misc-include-cleaner) bug in clang-tidy
    stack_t old_stack{nullptr};  // NOLINT(misc-include-cleaner) bug in clang-tidy
};

class CPUSpecialSetup {
    SigAltStackSetup ss;

public:
    CPUSpecialSetup() = default;
};
#else   // __linux__
class CPUSpecialSetup {
public:
    CPUSpecialSetup() = default;
};
#endif  // __linux__

}  // namespace ov::intel_cpu
