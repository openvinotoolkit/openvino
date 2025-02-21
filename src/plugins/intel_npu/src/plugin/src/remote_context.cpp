// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include "intel_npu/config/common.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

using namespace ov::intel_npu;

namespace {

template <typename Type>
std::optional<Type> extract_object(const ov::AnyMap& params, const ov::Property<Type>& p, bool isMandatory = true) {
    auto itrHandle = params.find(p.name());
    if (itrHandle == params.end()) {
        if (isMandatory) {
            OPENVINO_THROW("No parameter ", p.name(), " found in parameters map");
        }

        return std::nullopt;
    }

    return ov::Any(itrHandle->second).as<Type>();
}

}  // namespace

namespace intel_npu {

RemoteContextImpl::RemoteContextImpl(const std::shared_ptr<const NPUBackends>& backends, const Config& config)
    : _config(config),
      _device(backends->getDevice(_config.get<DEVICE_ID>())),
      _properties({l0_context(backends->getContext())}),
      _device_name("NPU") {
    if (_device == nullptr) {
        OPENVINO_THROW("Device is not available");
    }
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return _properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    if (params.empty()) {
        return _device->createRemoteTensor(get_this_shared_ptr(), type, shape, _config);
    }

    auto mem_type_object = extract_object(params, mem_type);

    TensorType tensor_type_object = TensorType::BINDED;
    void* mem_handle_object = nullptr;

    switch (*mem_type_object) {
    case MemType::L0_INTERNAL_BUF: {
        auto object = extract_object(params, tensor_type, false);
        if (object.has_value()) {
            tensor_type_object = *object;
        }
        break;
    }
    case MemType::SHARED_BUF: {
        auto object = extract_object(params, mem_handle);
        if (object.has_value()) {
            mem_handle_object = *object;
        }
        break;
    }
    default:
        OPENVINO_THROW("Unsupported shared object type ", *mem_type_object);
    }

    return _device->createRemoteTensor(get_this_shared_ptr(),
                                       type,
                                       shape,
                                       _config,
                                       tensor_type_object,
                                       *mem_type_object,
                                       mem_handle_object);
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return _device->createHostTensor(get_this_shared_ptr(), type, shape, _config);
}

const std::string& RemoteContextImpl::get_device_name() const {
    return _device_name;
}

std::shared_ptr<ov::IRemoteContext> RemoteContextImpl::get_this_shared_ptr() {
    return shared_from_this();
}

}  // namespace intel_npu
