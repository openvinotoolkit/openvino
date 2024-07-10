// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/utils/remote_tensor_type/remote_tensor_type.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

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

RemoteContextImpl::RemoteContextImpl(std::shared_ptr<const NPUBackends> backends, const Config& config)
    : _backends(backends),
      _config(config),
      _properties({ov::intel_npu::l0_context(backends->getContext())}),
      _device_name("NPU") {}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return _properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    auto device = _backends->getDevice(_config.get<DEVICE_ID>());
    if (device == nullptr) {
        OPENVINO_THROW("Device is not available");
    }

    if (params.empty()) {
        return device->createRemoteTensor(get_this_shared_ptr(), type, shape, _config);
    }

    auto mem_type = extract_object(params, ov::intel_npu::mem_type);

    RemoteMemoryType memory_type;
    RemoteTensorType tensor_type = RemoteTensorType::BINDED;
    void* mem = nullptr;

    if (!mem_type.has_value()) {
        OPENVINO_THROW("Shared object type should be set in parameter map");
    }

    switch (*mem_type) {
    case ov::intel_npu::MemType::L0_INTERNAL_BUF: {
        memory_type = RemoteMemoryType::L0_INTERNAL_BUF;
        auto object = extract_object(params, ov::intel_npu::tensor_type, false);
        if (object.has_value()) {
            switch (*object) {
            case ov::intel_npu::TensorType::INPUT:
                tensor_type = RemoteTensorType::INPUT;
                break;
            case ov::intel_npu::TensorType::OUTPUT:
                tensor_type = RemoteTensorType::OUTPUT;
                break;
            case ov::intel_npu::TensorType::BINDED:
                tensor_type = RemoteTensorType::BINDED;
                break;
            default:
                OPENVINO_THROW("Unknown tensor type");
            }
        }
        break;
    }
    case ov::intel_npu::MemType::SHARED_BUF: {
        memory_type = RemoteMemoryType::SHARED_BUF;
        auto object = extract_object(params, ov::intel_npu::mem_handle);
        if (object.has_value()) {
            mem = *object;
        }
        break;
    }
    default:
        OPENVINO_THROW("Unsupported shared object type ", *mem_type);
    }

    return device->createRemoteTensor(get_this_shared_ptr(), type, shape, _config, tensor_type, memory_type, mem);
}

const std::string& RemoteContextImpl::get_device_name() const {
    return _device_name;
}

std::shared_ptr<ov::IRemoteContext> RemoteContextImpl::get_this_shared_ptr() {
    return shared_from_this();
}

}  // namespace intel_npu
