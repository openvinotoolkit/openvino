// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/ie_so_loader.h"
#include "cpp/ie_memory_state.hpp"
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

void VariableState::Reset() {
    VARIABLE_CALL_STATEMENT(_impl->Reset());
}

std::string VariableState::GetName() const {
    VARIABLE_CALL_STATEMENT(return _impl->GetName());
}

Blob::CPtr VariableState::GetState() const {
    VARIABLE_CALL_STATEMENT(return _impl->GetState());
}

void VariableState::SetState(Blob::Ptr state) {
    VARIABLE_CALL_STATEMENT(_impl->SetState(state));
}

}  // namespace InferenceEngine
