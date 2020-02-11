// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <cpp_interfaces/impl/ie_memory_state_internal.hpp>
#include "gna_plugin.hpp"

namespace  GNAPluginNS {
namespace memory {
class GNAMemoryState : public InferenceEngine::MemoryStateInternal {
    std::shared_ptr<GNAPlugin> plg;
 public:
    using Ptr = InferenceEngine::MemoryStateInternal::Ptr;

    explicit GNAMemoryState(std::shared_ptr<GNAPlugin> plg)
        : InferenceEngine::MemoryStateInternal("GNAResetState"), plg(plg) {}
    void Reset() override {
        plg->Reset();
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
