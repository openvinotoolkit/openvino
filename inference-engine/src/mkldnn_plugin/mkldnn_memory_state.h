// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/impl/ie_memory_state_internal.hpp"
#include "mkldnn_memory.h"

#include <string>

namespace MKLDNNPlugin {

class MKLDNNVariableState : public InferenceEngine::IVariableStateInternal {
public:
    MKLDNNVariableState(std::string name, MKLDNNMemoryPtr storage) :
            name(name), storage(storage) {}

    std::string GetName() const override;
    void Reset() override;
    void SetState(InferenceEngine::Blob::Ptr newState) override;
    InferenceEngine::Blob::CPtr GetState() const override;

private:
    std::string name;
    MKLDNNMemoryPtr storage;
};

}  // namespace MKLDNNPlugin