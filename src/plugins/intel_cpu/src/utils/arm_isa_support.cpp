// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"  // OPENVINO_ARCH_* macros used by the guard below

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    include "arm_isa_support.h"
#    include "openvino/core/except.hpp"
#    include "openvino/runtime/system_conf.hpp"

namespace ov::intel_cpu {

bool hasArmISASupport(ArmISA isa) {
    switch (isa) {
    case ArmISA::ASIMD:
        return true;  // ARMv8-A baseline, always present
    case ArmISA::SVE:
        return with_cpu_sve();  // any SVE vector length
    case ArmISA::DOTPROD:
        return with_cpu_arm_dotprod();
    case ArmISA::I8MM:
        return with_cpu_arm_i8mm();
    }
    // An unhandled ArmISA must never be silently reported as supported: that could let an
    // executor emit instructions the core lacks. Fail loudly so a newly added ISA is wired up.
    OPENVINO_THROW("hasArmISASupport: unhandled ArmISA value");
}

}  // namespace ov::intel_cpu

#endif  // OPENVINO_ARCH_ARM || OPENVINO_ARCH_ARM64
