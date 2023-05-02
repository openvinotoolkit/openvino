// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<TransposeExecutorDesc>& getTransposeExecutorsList() {
    static std::vector<TransposeExecutorDesc> descs = {
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_DNNL(ExecutorType::Dnnl, std::make_shared<DNNLTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefTransposeExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov