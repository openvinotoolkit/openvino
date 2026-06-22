// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

bool hasHardwareSupport(const ov::element::Type& precision);
ov::element::Type defaultFloatPrecision();

// ARM ISA an executor's kernels require, declared in its supports() predicate.
// ASIMD is the ARM baseline; an executor needing only NEON declares nothing.
// An executor whose kernels use a higher ISA (e.g. SVE) requires it so that a core
// without it declines and the framework falls back to a baseline implementation.
enum class ArmISA : uint8_t { ASIMD, SVE, DOTPROD, I8MM };

// Whether the current core supports `isa`.
bool hasArmISASupport(ArmISA isa);

}  // namespace ov::intel_cpu
