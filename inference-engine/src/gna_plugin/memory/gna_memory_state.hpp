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
        auto const oldPrecision = memState->getTensorDesc().getPrecision();
        auto const oldByteSize = memState->byteSize();

        auto const newPrecision = newState->getTensorDesc().getPrecision();
        auto const newByteSize = newState->byteSize();

        if (newPrecision != InferenceEngine::Precision::FP32 && newPrecision != InferenceEngine::Precision::I16) {
            THROW_GNA_EXCEPTION << "SetState call failed. Allowed new state precisions: FP32 and I16";
        }

        if (oldPrecision == newPrecision) {
            if (oldByteSize == newByteSize) {
                std::memcpy(memState->buffer().as<uint8_t*>(),
                            newState->buffer().as<uint8_t*>(),
                            oldByteSize);
            } else {
                THROW_GNA_EXCEPTION << "SetState call failed. Precision is same but byte sizes don't match";
            }
        } else {
            switch (oldPrecision) {
            case InferenceEngine::Precision::FP32:
                THROW_GNA_EXCEPTION << "SetState call failed. When state precision is FP32 new state precision should be FP32";
            case InferenceEngine::Precision::I16: {
                if (newPrecision == InferenceEngine::Precision::FP32 && newByteSize / 2 == oldByteSize) {
                    ConvertToInt16(memState->buffer().as<int16_t*>(),
                        newState->buffer().as<float*>(),
                        newState->getTensorDesc().getDims()[0],
                        newState->getTensorDesc().getDims()[1],
                        scalefactor);
                } else {
                    THROW_GNA_EXCEPTION << "SetState call failed. Invalid precision / size";
                }
            }
            default:
                THROW_GNA_EXCEPTION << "SetState call failed. Current state precision is not supported";
            }
        }
    }

    InferenceEngine::Blob::CPtr GetLastState() const override {
        return memState;
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
