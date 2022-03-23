// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_remote_context.hpp"

#include <exception>

#include "any_copy.hpp"
#include "cpp_interfaces/interface/iremote_context.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/remote_context.hpp"

#define OV_REMOTE_CONTEXT_STATEMENT(...)                                     \
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized."); \
    type_check(*this);                                                       \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        throw ov::Exception(ex.what());                                      \
    } catch (...) {                                                          \
        OPENVINO_ASSERT(false, "Unexpected exception");                      \
    }

namespace ov {

void RemoteContext::type_check(const RemoteContext& remote_context,
                               const std::map<std::string, std::vector<std::string>>& type_info) {
    if (!type_info.empty()) {
        auto params = remote_context._impl->get_params();
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

RemoteContext::~RemoteContext() {
    _impl = {};
}

RemoteContext::RemoteContext(const IRemoteContext::Ptr& impl, const std::shared_ptr<void>& so) : _impl{impl}, _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized.");
}

RemoteContext::RemoteContext(const ie::RemoteContext::Ptr& impl, const std::shared_ptr<void>& so)
    : _impl{std::make_shared<IERemoteContext>(impl)},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized.");
}

std::string RemoteContext::get_device_name() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->get_device_name(););
}

RemoteTensor RemoteContext::create_tensor(const element::Type type, const Shape& shape, const AnyMap& params) {
    OV_REMOTE_CONTEXT_STATEMENT(return {_impl->create_tensor(type, shape, params), _so});
}

Tensor RemoteContext::create_host_tensor(const element::Type element_type, const Shape& shape) {
    OV_REMOTE_CONTEXT_STATEMENT(return {_impl->create_host_tensor(element_type, shape), _so};);
}

AnyMap RemoteContext::get_params() const {
    AnyMap paramMap;
    OV_REMOTE_CONTEXT_STATEMENT({
        for (auto&& param : _impl->get_params()) {
            paramMap.emplace(param.first, Any{param.second, _so});
        }
    });
    return paramMap;
}

}  // namespace ov
