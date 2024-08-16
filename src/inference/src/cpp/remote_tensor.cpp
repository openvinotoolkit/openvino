// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/remote_tensor.hpp"

#include <memory>

#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

RemoteTensor::RemoteTensor(const RemoteTensor& owner, const Coordinate& begin, const Coordinate& end) {
    _impl = make_tensor(std::dynamic_pointer_cast<ov::IRemoteTensor>(owner._impl), begin, end);
    _so = owner._so;
}

void RemoteTensor::type_check(const Tensor& tensor, const std::map<std::string, std::vector<std::string>>& type_info) {
    OPENVINO_ASSERT(tensor, "Could not check empty tensor type");
    auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(get_tensor_impl(tensor)._ptr);
    OPENVINO_ASSERT(remote_tensor, "Tensor is not remote.");
    if (!type_info.empty()) {
        auto remote_properties = remote_tensor->get_properties();
        for (auto&& type_info_value : type_info) {
            auto it_param = remote_properties.find(type_info_value.first);
            OPENVINO_ASSERT(it_param != remote_properties.end(),
                            "Parameter with key ",
                            type_info_value.first,
                            " not found");
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
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
    type_check(*this);
    auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(_impl);
    try {
        AnyMap paramMap;
        for (auto&& param : remote_tensor->get_properties()) {
            paramMap.emplace(param.first, Any{param.second, _so});
        }
        return paramMap;
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception");
    }
}

void RemoteTensor::copy_to(ov::Tensor& dst) const {
    auto remote_tensor_impl = std::dynamic_pointer_cast<ov::IRemoteTensor>(_impl);
    remote_tensor_impl->copy_to(get_tensor_impl(dst)._ptr);
}

void RemoteTensor::copy_from(const ov::Tensor& src) {
    auto remote_tensor_impl = std::dynamic_pointer_cast<ov::IRemoteTensor>(_impl);
    remote_tensor_impl->copy_from(get_tensor_impl(src)._ptr);
}

std::string RemoteTensor::get_device_name() const {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
    auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(_impl);
    OPENVINO_ASSERT(remote_tensor, "Tensor is not remote.");
    type_check(*this);
    try {
        return remote_tensor->get_device_name();
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception");
    }
}
}  // namespace ov
