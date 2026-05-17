// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/core/Error.h>

#include "utils/precision_support.h"

namespace ov::intel_cpu {

inline bool mayUseAclGemmBasedExecutor() {
#if defined(OPENVINO_ARCH_ARM64)
    return hasArmSVESupport();
#else
    return true;
#endif
}

inline arm_compute::Status aclGemmBasedExecutorUnsupportedStatus() {
    return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                               "ACL GEMM-based executor requires ARM SVE runtime support");
}

}  // namespace ov::intel_cpu
