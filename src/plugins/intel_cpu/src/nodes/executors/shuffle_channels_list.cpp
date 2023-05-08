// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<ShuffleChannelsExecutorDesc>& getShuffleChannelsExecutorsList() {
    static std::vector<ShuffleChannelsExecutorDesc> descs = {
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLShuffleChannelsExecutorBuilder>())
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<CommonShuffleChannelsExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov