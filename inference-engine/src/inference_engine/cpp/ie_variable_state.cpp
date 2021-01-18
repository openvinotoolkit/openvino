// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "ie_imemory_state.hpp"
#include "cpp/ie_memory_state.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

VariableState::VariableState(IVariableState::Ptr pState, details::SharedObjectLoader::Ptr plg) : actual(pState), plugin(plg) {
    if (actual == nullptr) {
        THROW_IE_EXCEPTION << "VariableState wrapper was not initialized.";
    }
}

Blob::CPtr VariableState::GetLastState() const {
    return GetState();
}

IE_SUPPRESS_DEPRECATED_END

void VariableState::Reset() {
    CALL_STATUS_FNC_NO_ARGS(Reset);
}

std::string VariableState::GetName() const {
    char name[256];
    CALL_STATUS_FNC(GetName, name, sizeof(name));
    return name;
}

Blob::CPtr VariableState::GetState() const {
    Blob::CPtr stateBlob;
    CALL_STATUS_FNC(GetState, stateBlob);
    return stateBlob;
}

void VariableState::SetState(Blob::Ptr state) {
    CALL_STATUS_FNC(SetState, state);
}

}  // namespace InferenceEngine