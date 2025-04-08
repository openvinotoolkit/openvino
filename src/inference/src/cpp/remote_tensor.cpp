// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/remote_tensor.hpp"

#include <memory>

#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

#define OV_REMOTE_TENSOR_STATEMENT(remote_tensor, ...)                        \
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");         \
    auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(_impl); \
    OPENVINO_ASSERT(remote_tensor, "Tensor is not remote.");                  \
    try {                                                                     \
        __VA_ARGS__;                                                          \
    } catch (const std::exception& ex) {                                      \
        OPENVINO_THROW(ex.what());                                            \
    } catch (...) {                                                           \
        OPENVINO_THROW("Unexpected exception");                               \
    }

RemoteTensor::RemoteTensor(const RemoteTensor& owner, const Coordinate& begin, const Coordinate& end) {
    OPENVINO_ASSERT(get_tensor_impl(owner)._ptr, "Cannot create RoiRemoteTensor on top of empty tensor");
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
    auto get_params_impl = [](const std::shared_ptr<ov::IRemoteTensor>& remote_tensor,
                              const std::shared_ptr<void>& so) {
        ov::AnyMap params_map;
        for (auto&& param : remote_tensor->get_properties()) {
            params_map.emplace(param.first, Any{param.second, so});
        }
        return params_map;
    };

    OV_REMOTE_TENSOR_STATEMENT(remote_tensor, return get_params_impl(remote_tensor, _so));
}

void RemoteTensor::copy_to(ov::Tensor& dst) const {
    OV_REMOTE_TENSOR_STATEMENT(remote_tensor, remote_tensor->copy_to(get_tensor_impl(dst)._ptr));
}

void RemoteTensor::copy_from(const ov::Tensor& src) {
    OV_REMOTE_TENSOR_STATEMENT(remote_tensor, remote_tensor->copy_from(get_tensor_impl(src)._ptr));
}

std::string RemoteTensor::get_device_name() const {
    OV_REMOTE_TENSOR_STATEMENT(remote_tensor, return remote_tensor->get_device_name());
}
}  // namespace ov
