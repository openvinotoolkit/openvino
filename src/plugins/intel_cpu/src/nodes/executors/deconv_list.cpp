// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv_list.hpp"

#include <vector>

#include "utils/arch_macros.h"
#if defined(OPENVINO_ARCH_ARM64)
#    include "nodes/executors/aarch64/jit_deconv3d.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#    include <memory>

#    include "nodes/executors/acl/acl_deconv.hpp"
#    include "nodes/executors/executor.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<DeconvExecutorDesc>& getDeconvExecutorsList() {
    static std::vector<DeconvExecutorDesc> descs = {
        // Prefer ACL builder first for stability/perf; fallback to AArch64 JIT if ACL not supported
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclDeconvExecutorBuilder>())
            OV_CPU_INSTANCE_ARM64(ExecutorType::Jit, std::make_shared<AArch64JitDeconvExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
