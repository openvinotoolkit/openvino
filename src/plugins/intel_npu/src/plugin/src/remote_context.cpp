// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include "intel_npu/config/common.hpp"

using namespace ov::intel_npu;

namespace {

template <typename Type>
std::optional<Type> extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    if (itrHandle == params.end()) {
        return std::nullopt;
    }

    return ov::Any(itrHandle->second).as<Type>();
}

}  // namespace

namespace intel_npu {

RemoteContextImpl::RemoteContextImpl(const ov::SoPtr<IEngineBackend>& engineBackend,
                                     const Config& config,
                                     const ov::AnyMap& remote_properties)
    : _config(config),
      _device_name("NPU") {
    if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
        OPENVINO_THROW("Level zero backend is not found!");
    }

    _device = engineBackend->getDevice(config.get<DEVICE_ID>());
    if (_device == nullptr) {
        OPENVINO_THROW("Device is not available!");
    }

    _properties = {l0_context(engineBackend->getContext())};

    if (!remote_properties.empty()) {
        _mem_type_object = extract_object(remote_properties, mem_type);
        _tensor_type_object = extract_object(remote_properties, tensor_type);
        _mem_handle_object = extract_object(remote_properties, mem_handle);
    }
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return _properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    // Local remote properties
    std::optional<ov::intel_npu::MemType> mem_type_object = std::nullopt;
    std::optional<ov::intel_npu::TensorType> tensor_type_object = std::nullopt;
    std::optional<void*> mem_handle_object = std::nullopt;

    if (!params.empty()) {
        // Save local remote properties.
        mem_type_object = extract_object(params, mem_type);
        tensor_type_object = extract_object(params, tensor_type);
        mem_handle_object = extract_object(params, mem_handle);
    }

    // Merge local remote properties with global remote properties.
    if (!mem_type_object.has_value() && _mem_type_object.has_value()) {
        mem_type_object = _mem_type_object;
    }
    if (!tensor_type_object.has_value() && _tensor_type_object.has_value()) {
        tensor_type_object = _tensor_type_object;
    }
    if (!mem_handle_object.has_value() && _mem_handle_object.has_value()) {
        mem_handle_object = _mem_handle_object;
    }

    // Mem_type shall be set if any other property is set.
    if (!mem_type_object.has_value() && (mem_handle_object.has_value() || tensor_type_object.has_value())) {
        OPENVINO_THROW("Parameter ", mem_type.name(), " must be set");
    }

    if (!mem_type_object.has_value()) {
        return _device->createRemoteTensor(get_this_shared_ptr(), type, shape, _config);
    }

    // Mem_handle shall be set if mem_type is a shared memory type.
    if (mem_type_object.value() == MemType::SHARED_BUF && !mem_handle_object.has_value()) {
        OPENVINO_THROW("No parameter ", mem_handle.name(), " found in parameters map");
    }

    return _device->createRemoteTensor(get_this_shared_ptr(),
                                       type,
                                       shape,
                                       _config,
                                       tensor_type_object.value_or(ov::intel_npu::TensorType::BINDED),
                                       mem_type_object.value_or(ov::intel_npu::MemType::L0_INTERNAL_BUF),
                                       mem_handle_object.value_or(nullptr));
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return _device->createHostTensor(get_this_shared_ptr(),
                                     type,
                                     shape,
                                     _config,
                                     _tensor_type_object.value_or(ov::intel_npu::TensorType::BINDED));
}

const std::string& RemoteContextImpl::get_device_name() const {
    return _device_name;
}

std::shared_ptr<ov::IRemoteContext> RemoteContextImpl::get_this_shared_ptr() {
    return shared_from_this();
}

}  // namespace intel_npu
