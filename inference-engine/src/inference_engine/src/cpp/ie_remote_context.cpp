// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_remote_context.hpp"

#include <exception>

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
namespace runtime {

void RemoteContext::type_check(const RemoteContext& tensor,
                               const std::map<std::string, std::vector<std::string>>& type_info) {
    auto remote_impl = dynamic_cast<const ie::RemoteContext*>(tensor._impl.get());
    OPENVINO_ASSERT(remote_impl != nullptr, "Context was not initialized using remote implementation");
    if (!type_info.empty()) {
        auto params = remote_impl->getParams();
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

RemoteContext::RemoteContext(const std::shared_ptr<void>& so, const ie::RemoteContext::Ptr& impl)
    : _so{so},
      _impl{impl} {
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized.");
}

std::string RemoteContext::get_device_name() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getDeviceName());
}

RemoteTensor RemoteContext::create_tensor(const element::Type& element_type,
                                          const Shape& shape,
                                          const ie::ParamMap& params) {
    OV_REMOTE_CONTEXT_STATEMENT({
        auto blob = _impl->CreateBlob(
            {ie::details::convertPrecision(element_type), shape, ie::TensorDesc::getLayoutByRank(shape.size())},
            params);
        blob->allocate();
        return {_so, blob};
    });
}

ie::ParamMap RemoteContext::get_params() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getParams());
}

}  // namespace runtime
}  // namespace ov
