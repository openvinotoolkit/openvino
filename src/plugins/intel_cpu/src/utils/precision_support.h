// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::intel_cpu {

bool hasHardwareSupport(const ov::element::Type& precision);
ov::element::Type defaultFloatPrecision();
bool hasIntDotProductSupport();
bool hasInt8MMSupport();

// The ARM ISA gate below is only referenced by the ARM executors (ACL / KleidiAI),
// which are compiled exclusively on ARM, so it is declared for ARM targets only.
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)

// Minimum ARM ISA an executor's kernels require, to be declared in its supports()
// predicate. ASIMD is the ARMv8-A baseline (present on every AArch64 core), so it
// is the implicit default: an executor that needs only baseline NEON declares
// nothing and is never gated. Only executors whose kernels emit above-baseline
// instructions (e.g. SVE) should require the corresponding ISA, so that on a core
// lacking it the executor declines cleanly and the framework falls back to a
// baseline implementation instead of executing an illegal instruction.
//
// Usage in an executor's supports() predicate:
//     VERIFY(hasArmISASupport(ArmISA::SVE), UNSUPPORTED_ISA);
enum class ArmISA : uint8_t { ASIMD, SVE, DOTPROD, I8MM };

// Returns whether the current core supports `isa`. ASIMD is always available on
// AArch64; on 32-bit ARM the gate is permissive (NEON is the baseline there too).
bool hasArmISASupport(ArmISA isa);

#endif  // OPENVINO_ARCH_ARM || OPENVINO_ARCH_ARM64

}  // namespace ov::intel_cpu
