// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/ivariable_state.hpp"

#include "openvino/itensor.hpp"

namespace ov {

IVariableState::IVariableState(const std::string& name_) : name{name_} {}

IVariableState::~IVariableState() = default;

std::string IVariableState::get_name() const {
    return name;
}

void IVariableState::reset() {
    OPENVINO_UNREACHABLE("Not implemented");
}

void IVariableState::set_state(const std::shared_ptr<ITensor>& new_state) {
    state = new_state;
}

std::shared_ptr<ITensor> IVariableState::get_state() const {
    return state;
}

IEVariableState::IEVariableState(const ie::IVariableStateInternal::Ptr& impl_)
    : IVariableState{impl_->GetName()},
      impl{impl_} {}

std::string IEVariableState::get_name() const {
    return impl->GetName();
}

void IEVariableState::reset() {
    impl->Reset();
}

void IEVariableState::set_state(const std::shared_ptr<ITensor>& new_state) {
    impl->SetState(tensor_to_blob(new_state));
}

std::shared_ptr<ITensor> IEVariableState::get_state() const {
    return blob_to_tensor(std::const_pointer_cast<ie::Blob>(impl->GetState()));
}
}  // namespace ov
