// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_state.hpp"

namespace  GNAPluginNS {

namespace memory {

    std::string GNAMemoryState::GetName() const {
        return name;
    }

    void GNAMemoryState::Reset() {
        state.Reset();
    }

    void GNAMemoryState::SetState(InferenceEngine::Blob::Ptr newState) {
        auto data_ptr = newState->cbuffer().as<void*>();
        auto data_size = newState->byteSize();

        state.gna_ptr = data_ptr;
        state.reserved_size = data_size;
    }

    InferenceEngine::Blob::CPtr GNAMemoryState::GetLastState() const {
        THROW_GNA_EXCEPTION << "GetLastState method is not yet implemented for GNAMemoryState";
        return nullptr;
    }
}  // namespace memory
}  // namespace GNAPluginNS
