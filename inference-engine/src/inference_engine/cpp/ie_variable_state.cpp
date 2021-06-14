// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_so_loader.h"
#include "cpp/ie_memory_state.hpp"
#include "ie_imemory_state.hpp"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "exception2status.hpp"

#define VARIABLE_CALL_STATEMENT(...)                                                               \
    if (_impl == nullptr) IE_THROW(NotAllocated) << "VariableState was not initialized.";          \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } catch(...) {details::Rethrow();}

namespace InferenceEngine {

VariableState::VariableState(const details::SharedObjectLoader& so,
                             const IVariableStateInternal::Ptr& impl)
    : _so(so), _impl(impl) {
    if (_impl == nullptr) IE_THROW() << "VariableState was not initialized.";
}

IE_SUPPRESS_DEPRECATED_START

VariableState::VariableState(std::shared_ptr<IVariableState> state,
                             std::shared_ptr<details::SharedObjectLoader> splg)
    : _so(), _impl(), actual(state) {
    if (splg) {
        _so = *splg;
    }

    //  plg can be null, but not the actual
    if (actual == nullptr)
        IE_THROW(NotAllocated) << "VariableState was not initialized.";
}

Blob::CPtr VariableState::GetLastState() const {
    return GetState();
}

void VariableState::Reset() {
    if (actual) {
        CALL_STATUS_FNC_NO_ARGS(Reset);
        return;
    }

    VARIABLE_CALL_STATEMENT(_impl->Reset());
}

std::string VariableState::GetName() const {
    if (actual) {
        char name[256];
        CALL_STATUS_FNC(GetName, name, sizeof(name));
        return name;
    }

    VARIABLE_CALL_STATEMENT(return _impl->GetName());
}

Blob::CPtr VariableState::GetState() const {
    if (actual) {
        Blob::CPtr stateBlob;
        CALL_STATUS_FNC(GetState, stateBlob);
        return stateBlob;
    }

    VARIABLE_CALL_STATEMENT(return _impl->GetState());
}

void VariableState::SetState(Blob::Ptr state) {
    if (actual) {
        CALL_STATUS_FNC(SetState, state);
        return;
    }

    VARIABLE_CALL_STATEMENT(_impl->SetState(state));
}

}  // namespace InferenceEngine
