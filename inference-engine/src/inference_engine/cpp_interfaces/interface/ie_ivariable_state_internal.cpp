// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>

namespace InferenceEngine {
IVariableStateInternal::IVariableStateInternal(const std::string& name_) : name{name_} {}

std::string IVariableStateInternal::GetName() const {
    return name;
}

void IVariableStateInternal::Reset() {
    IE_THROW(NotImplemented);
}

void IVariableStateInternal::SetState(const Blob::Ptr& newState)  {
    state = newState;
}

Blob::CPtr IVariableStateInternal::GetState() const {
    return state;
}

}  // namespace InferenceEngine
