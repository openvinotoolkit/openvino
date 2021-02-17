// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/impl/ie_variable_state_internal.hpp"
#include "blob_factory.hpp"
#include "mkldnn_memory.h"
#include "nodes/common/cpu_memcpy.h"

#include <string>

namespace MKLDNNPlugin {

class MKLDNNVariableState : public InferenceEngine::IVariableStateInternal {
public:
    MKLDNNVariableState(std::string name, MKLDNNMemoryPtr storage) :
            name(name) {
        this->storage = make_blob_with_precision(MKLDNNMemoryDesc(storage->GetDescriptor()));
        this->storage->allocate();
        cpu_memcpy(this->storage->buffer(), storage->GetData(), storage->GetSize());
    }

    std::string GetName() const override;
    void Reset() override;
    void SetState(InferenceEngine::Blob::Ptr newState) override;
    InferenceEngine::Blob::CPtr GetState() const override;

private:
    std::string name;
    InferenceEngine::Blob::Ptr storage;
};

}  // namespace MKLDNNPlugin