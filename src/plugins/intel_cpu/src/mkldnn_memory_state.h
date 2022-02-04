// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "blob_factory.hpp"
#include "mkldnn_memory.h"
#include "nodes/common/cpu_memcpy.h"
#include "memory_desc/cpu_memory_desc_utils.h"

#include <string>

namespace MKLDNNPlugin {

class MKLDNNVariableState : public InferenceEngine::IVariableStateInternal {
public:
    MKLDNNVariableState(std::string name, MKLDNNMemoryPtr storage) :
            InferenceEngine::IVariableStateInternal{name} {
        state = make_blob_with_precision(MemoryDescUtils::convertToTensorDesc(storage->getDesc()));
        state->allocate();
        cpu_memcpy(state->buffer(), storage->GetData(), storage->GetSize());
    }

    void Reset() override;
};

}  // namespace MKLDNNPlugin