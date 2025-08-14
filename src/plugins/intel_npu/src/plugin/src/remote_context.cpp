// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

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
                                     const ov::AnyMap& remote_properties)
    : _device_name("NPU") {
    if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
        OPENVINO_THROW("Level zero backend is not found!");
    }

    _init_structs = engineBackend->getInitStructs();

    _properties = {l0_context(engineBackend->getContext())};

    if (!remote_properties.empty()) {
        _mem_type_object = extract_object(remote_properties, mem_type);
        _tensor_type_object = extract_object(remote_properties, tensor_type);
        _mem_handle_object = extract_object(remote_properties, mem_handle);
        _file_descriptor_object = extract_object(remote_properties, file_descriptor);
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
    std::optional<ov::intel_npu::FileDescriptor> file_descriptor_object = std::nullopt;

    if (!params.empty()) {
        // Save local remote properties.
        mem_type_object = extract_object(params, mem_type);
        tensor_type_object = extract_object(params, tensor_type);
        mem_handle_object = extract_object(params, mem_handle);
        file_descriptor_object = extract_object(params, file_descriptor);
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
    if (!file_descriptor_object.has_value() && _file_descriptor_object.has_value()) {
        file_descriptor_object = _file_descriptor_object;
    }

    // Mem_type shall be set if any other property is set.
    if (!mem_type_object.has_value() &&
        (mem_handle_object.has_value() || tensor_type_object.has_value() || file_descriptor_object.has_value())) {
        OPENVINO_THROW("Parameter ", mem_type.name(), " must be set");
    }

    if (!mem_type_object.has_value()) {
        return {std::make_shared<ZeroRemoteTensor>(get_this_shared_ptr(), _init_structs, type, shape)};
    }

    // Mem_handle shall be set if mem_type is a shared memory type.
    if (mem_type_object.value() == MemType::SHARED_BUF && !mem_handle_object.has_value()) {
        OPENVINO_THROW("No parameter ", mem_handle.name(), " found in parameters map");
    }

    return {std::make_shared<ZeroRemoteTensor>(get_this_shared_ptr(),
                                               _init_structs,
                                               type,
                                               shape,
                                               tensor_type_object.value_or(ov::intel_npu::TensorType::BINDED),
                                               mem_type_object.value_or(ov::intel_npu::MemType::L0_INTERNAL_BUF),
                                               mem_handle_object.value_or(nullptr),
                                               file_descriptor_object)};
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return {std::make_shared<ZeroHostTensor>(get_this_shared_ptr(),
                                             _init_structs,
                                             type,
                                             shape,
                                             _tensor_type_object.value_or(ov::intel_npu::TensorType::BINDED))};
}

const std::string& RemoteContextImpl::get_device_name() const {
    return _device_name;
}

std::shared_ptr<ov::IRemoteContext> RemoteContextImpl::get_this_shared_ptr() {
    return shared_from_this();
}

}  // namespace intel_npu
