// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_memory_state.hpp"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/variable_state.hpp"

#define VARIABLE_CALL_STATEMENT(...)                                    \
    if (_impl == nullptr)                                               \
        IE_THROW(NotAllocated) << "VariableState was not initialized."; \
    try {                                                               \
        __VA_ARGS__;                                                    \
    } catch (...) {                                                     \
        ::InferenceEngine::details::Rethrow();                          \
    }

#define OV_VARIABLE_CALL_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "VariableState was not initialized."); \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        throw ov::Exception(ex.what());                                      \
    } catch (...) {                                                          \
        OPENVINO_ASSERT(false, "Unexpected exception");                      \
    }

namespace InferenceEngine {

VariableState::~VariableState() {
    _impl = {};
}

VariableState::VariableState(const IVariableStateInternal::Ptr& impl, const std::shared_ptr<void>& so)
    : _impl(impl),
      _so(so) {
    if (_impl == nullptr)
        IE_THROW() << "VariableState was not initialized.";
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

namespace ov {

VariableState::~VariableState() {
    _impl = {};
}

VariableState::VariableState(const IVariableState::Ptr& impl, const std::shared_ptr<void>& so) : _impl{impl}, _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "VariableState was not initialized.");
}

VariableState::VariableState(const ie::IVariableStateInternal::Ptr& impl, const std::shared_ptr<void>& so)
    : _impl{std::make_shared<IEVariableState>(impl)},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "VariableState was not initialized.");
}

void VariableState::reset() {
    OV_VARIABLE_CALL_STATEMENT(_impl->reset());
}

std::string VariableState::get_name() const {
    OV_VARIABLE_CALL_STATEMENT(return _impl->get_name());
}

Tensor VariableState::get_state() const {
    OV_VARIABLE_CALL_STATEMENT(return {_impl->get_state(), _so});
}

void VariableState::set_state(const Tensor& state) {
    OV_VARIABLE_CALL_STATEMENT(_impl->set_state(state._impl));
}

}  // namespace ov
