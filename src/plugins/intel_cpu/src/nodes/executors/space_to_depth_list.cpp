// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<SpaceToDepthExecutorDesc>& getSpaceToDepthExecutorsList() {
    static std::vector<SpaceToDepthExecutorDesc> descs = {
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLSpaceToDepthExecutorBuilder>())
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<CommonSpaceToDepthExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov