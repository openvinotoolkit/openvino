// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <cpp_interfaces/impl/ie_memory_state_internal.hpp>
#include <ie_blob.h>
#include "gna_plugin.hpp"
#include "preprocessing.hpp"

namespace  GNAPluginNS {
namespace memory {
class GNAMemoryState : public InferenceEngine::MemoryStateInternal {
    std::string                 stateName;
    InferenceEngine::Blob::Ptr  memState;
    float                       scalefactor;

 public:
    using Ptr = InferenceEngine::MemoryStateInternal::Ptr;

    explicit GNAMemoryState(std::string name,
                            InferenceEngine::Blob::Ptr state,
                            float scale_factor)
        : InferenceEngine::MemoryStateInternal(name), stateName(name),
            memState(state), scalefactor(scale_factor) {}

    void Reset() override {
        std::memset(memState->buffer().as<int16_t*>(), 0, memState->byteSize());
    }

    void SetState(InferenceEngine::Blob::Ptr newState) override {
        if (newState->getTensorDesc().getDims().size() != 2) {
            THROW_GNA_EXCEPTION << "SetState failed for blob dimensions > 2";
        }

        if ((newState->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) &&
            (newState->byteSize()/2 == memState->byteSize())) {
            ConvertToInt16(memState->buffer().as<int16_t*>(),
                            newState->buffer().as<float*>(),
                            newState->getTensorDesc().getDims()[0],
                            newState->getTensorDesc().getDims()[1],
                            scalefactor);
        } else if ((newState->getTensorDesc().getPrecision() == InferenceEngine::Precision::I16) &&
                    (newState->byteSize() == memState->byteSize())) {
            std::memcpy(memState->buffer().as<uint8_t*>(),
                        newState->buffer().as<uint8_t*>(),
                        newState->byteSize());
        } else {
            THROW_GNA_EXCEPTION << "SetState call failed. Invalid precision / size";
        }
    }

    InferenceEngine::Blob::CPtr GetLastState() const override {
        return memState;
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
