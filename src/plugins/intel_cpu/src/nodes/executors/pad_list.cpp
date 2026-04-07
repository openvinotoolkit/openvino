// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad_list.hpp"

#include <vector>

#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include <memory>

#    include "nodes/executors/acl/acl_pad.hpp"
#    include "nodes/executors/executor.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<PadExecutorDesc>& getPadExecutorsList() {
    static std::vector<PadExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclPadExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu