// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_so_loader.h"
#include "cpp/ie_memory_state.hpp"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "exception2status.hpp"

#define VARIABLE_CALL_STATEMENT(...)                                                               \
    if (_ptr == nullptr) IE_THROW() << "VariableState was not initialized.";                       \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {                                       \
        IE_THROW() << ex.what();                                                                   \
    } catch (...) {                                                                                \
        IE_THROW(Unexpected);                                                                      \
    }

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

Blob::CPtr VariableState::GetLastState() const {
    return GetState();
}

IE_SUPPRESS_DEPRECATED_END

void VariableState::Reset() {
    VARIABLE_CALL_STATEMENT(_ptr->Reset());
}

std::string VariableState::GetName() const {
    VARIABLE_CALL_STATEMENT(return _ptr->GetName());
}

Blob::CPtr VariableState::GetState() const {
    VARIABLE_CALL_STATEMENT(return _ptr->GetState());
}

void VariableState::SetState(Blob::Ptr state) {
    VARIABLE_CALL_STATEMENT(_ptr->SetState(state));
}

}  // namespace InferenceEngine
