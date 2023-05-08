// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<NormalizeL2ExecutorDesc>& getNormalizeL2ExecutorsList() {
    static std::vector<NormalizeL2ExecutorDesc> descs = {
        OV_CPU_INSTANCE_X64(ExecutorType::x64, std::make_shared<JitNormalizeL2ExecutorBuilder>())
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLNormalizeL2ExecutorBuilder>())
        OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefNormalizeL2ExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov