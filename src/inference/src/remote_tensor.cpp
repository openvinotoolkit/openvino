// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/remote_tensor.hpp"

#include "any_copy.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"

namespace ov {

void RemoteTensor::type_check(const Tensor& tensor, const std::map<std::string, std::vector<std::string>>& type_info) {
    OPENVINO_ASSERT(tensor, "Could not check empty tensor type");
    auto remote_tensor = static_cast<const RemoteTensor*>(&tensor);
    auto remote_impl = dynamic_cast<ie::RemoteBlob*>(remote_tensor->_impl.get());
    OPENVINO_ASSERT(remote_impl != nullptr, "Tensor was not initialized using remote implementation");
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

AnyMap RemoteTensor::get_params() const {
    OPENVINO_ASSERT(_impl != nullptr, "Remote tensor was not initialized.");
    type_check(*this);
    auto remote_impl = static_cast<ie::RemoteBlob*>(_impl.get());
    try {
        AnyMap paramMap;
        for (auto&& param : remote_impl->getParams()) {
            paramMap.emplace(param.first, Any{param.second, _so});
        }
        return paramMap;
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

std::string RemoteTensor::get_device_name() const {
    OPENVINO_ASSERT(_impl != nullptr, "Remote tensor was not initialized.");
    auto remote_impl = static_cast<ie::RemoteBlob*>(_impl.get());
    type_check(*this);
    try {
        return remote_impl->getDeviceName();
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}
}  // namespace ov
