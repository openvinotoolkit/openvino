// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/visibility.hpp"

namespace ov::intel_cpu {

// ARM-only: referenced solely by the ARM executors (ACL / KleidiAI). Not defined on x86_64 / RISC-V.
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)

// ISA an ARM executor requires in its supports() predicate. ASIMD is the baseline (declare
// nothing); requiring a higher ISA (e.g. SVE) makes the executor decline on incapable cores.
enum class ArmISA : uint8_t { ASIMD, SVE, DOTPROD, I8MM };

bool hasArmISASupport(ArmISA isa);

#endif  // OPENVINO_ARCH_ARM || OPENVINO_ARCH_ARM64

}  // namespace ov::intel_cpu
