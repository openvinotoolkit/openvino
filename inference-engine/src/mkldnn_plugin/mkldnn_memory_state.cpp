// Copyright (C) 2018-2020 Intel Corporation
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
    storage->FillZero();
}

void  MKLDNNVariableState::SetState(Blob::Ptr newState) {
    auto prec = newState->getTensorDesc().getPrecision();
    auto data_type = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
    auto data_layout = MKLDNNMemory::Convert(newState->getTensorDesc().getLayout());
    auto data_ptr = newState->cbuffer().as<void*>();
    auto data_size = newState->byteSize();

    storage->SetData(data_type, data_layout, data_ptr, data_size);
}

InferenceEngine::Blob::CPtr MKLDNNVariableState::GetState() const {
    auto result_blob = make_blob_with_precision(MKLDNNMemoryDesc(storage->GetDescriptor()));
    result_blob->allocate();
    std::memcpy(result_blob->buffer(), storage->GetData(), storage->GetSize());
    return result_blob;
}

}  // namespace MKLDNNPlugin
