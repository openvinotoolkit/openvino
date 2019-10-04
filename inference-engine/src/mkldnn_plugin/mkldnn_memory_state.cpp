// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory_state.h"
#include "mkldnn_extension_utils.h"

using namespace InferenceEngine;

namespace MKLDNNPlugin {

std::string  MKLDNNMemoryState::GetName() const {
    return name;
}

void  MKLDNNMemoryState::Reset() {
    storage->FillZero();
}

void  MKLDNNMemoryState::SetState(Blob::Ptr newState) {
    auto prec = newState->getTensorDesc().getPrecision();
    auto data_type = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
    auto data_layout = MKLDNNMemory::Convert(newState->getTensorDesc().getLayout());
    auto data_ptr = newState->cbuffer().as<void*>();
    auto data_size = newState->byteSize();

    storage->SetData(data_type, data_layout, data_ptr, data_size);
}

InferenceEngine::Blob::CPtr MKLDNNMemoryState::GetLastState() const {
    THROW_IE_EXCEPTION << "GetLastState method is not implemented for MemoryState";
    return nullptr;
}

}  // namespace MKLDNNPlugin