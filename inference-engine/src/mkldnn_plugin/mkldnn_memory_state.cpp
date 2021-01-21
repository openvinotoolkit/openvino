// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory_state.h"
#include "mkldnn_extension_utils.h"
#include "blob_factory.hpp"

using namespace InferenceEngine;

namespace MKLDNNPlugin {

std::string  MKLDNNVariableState::GetName() const {
    return name;
}

void  MKLDNNVariableState::Reset() {
    std::memset(this->storage->buffer(), 0, storage->byteSize());
}

void  MKLDNNVariableState::SetState(Blob::Ptr newState) {
    storage = newState;
}

InferenceEngine::Blob::CPtr MKLDNNVariableState::GetState() const {
    return storage;
}

}  // namespace MKLDNNPlugin
