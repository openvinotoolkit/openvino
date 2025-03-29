// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/variable_state.hpp"

#include "openvino/core/except.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

#define OV_VARIABLE_CALL_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "VariableState was not initialized."); \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        OPENVINO_THROW(ex.what());                                           \
    } catch (...) {                                                          \
        OPENVINO_THROW("Unexpected exception");                              \
    }

namespace ov {

VariableState::~VariableState() {
    _impl = {};
}

VariableState::VariableState(const std::shared_ptr<ov::IVariableState>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
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
    OV_VARIABLE_CALL_STATEMENT({
        auto tensor = _impl->get_state();
        return make_tensor(tensor);
    });
}

void VariableState::set_state(const Tensor& state) {
    OV_VARIABLE_CALL_STATEMENT(_impl->set_state(get_tensor_impl(state)));
}

}  // namespace ov
