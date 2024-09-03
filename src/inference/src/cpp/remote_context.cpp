// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/remote_context.hpp"

#include <exception>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"

#define OV_REMOTE_CONTEXT_STATEMENT(...)                                     \
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized."); \
    type_check(*this);                                                       \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        OPENVINO_THROW(ex.what());                                           \
    } catch (...) {                                                          \
        OPENVINO_THROW("Unexpected exception");                              \
    }

namespace ov {

void RemoteContext::type_check(const RemoteContext& context,
                               const std::map<std::string, std::vector<std::string>>& type_info) {
    auto remote_impl = context._impl;
    OPENVINO_ASSERT(remote_impl != nullptr, "Context was not initialized using remote implementation");
    if (!type_info.empty()) {
        auto params = remote_impl->get_property();
        for (auto&& type_info_value : type_info) {
            auto it_param = params.find(type_info_value.first);
            OPENVINO_ASSERT(it_param != params.end(), "Parameter with key ", type_info_value.first, " not found");
            if (!type_info_value.second.empty()) {
                auto param_value = it_param->second.as<std::string>();
                auto param_found = std::any_of(type_info_value.second.begin(),
                                               type_info_value.second.end(),
                                               [&](const std::string& param) {
                                                   return param == param_value;
                                               });
                OPENVINO_ASSERT(param_found, "Unexpected parameter value ", param_value);
            }
        }
    }
}

RemoteContext::operator bool() const noexcept {
    return (!!_impl);
}

RemoteContext::~RemoteContext() {
    _impl = {};
}

RemoteContext::RemoteContext(const std::shared_ptr<ov::IRemoteContext>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized.");
}

std::string RemoteContext::get_device_name() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->get_device_name());
}

RemoteTensor RemoteContext::create_tensor(const element::Type& type, const Shape& shape, const AnyMap& params) {
    OV_REMOTE_CONTEXT_STATEMENT({
        auto tensor = _impl->create_tensor(type, shape, params);
        if (!tensor._so)
            tensor._so = _so;
        return make_tensor(tensor).as<ov::RemoteTensor>();
    });
}

Tensor RemoteContext::create_host_tensor(const element::Type element_type, const Shape& shape) {
    OV_REMOTE_CONTEXT_STATEMENT({
        auto tensor = _impl->create_host_tensor(element_type, shape);
        if (!tensor._so)
            tensor._so = _so;
        return make_tensor(tensor);
    });
}

AnyMap RemoteContext::get_params() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->get_property());
}

}  // namespace ov
