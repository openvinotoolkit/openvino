// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include "intel_npu/config/common.hpp"

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

RemoteContextImpl::RemoteContextImpl(const std::shared_ptr<const NPUBackends>& backends,
                                     const Config& config,
                                     const ov::AnyMap& remote_properties)
    : _config(config),
      _device(backends->getDevice(_config.get<DEVICE_ID>())),
      _properties({l0_context(backends->getContext())}),
      _device_name("NPU") {
    if (_device == nullptr) {
        OPENVINO_THROW("Device is not available");
    }

    if (!remote_properties.empty()) {
        _mem_type = extract_object(remote_properties, mem_type);

        switch (*_mem_type) {
        case MemType::L0_INTERNAL_BUF: {
            auto object = extract_object(remote_properties, tensor_type, false);
            if (object.has_value()) {
                _tensor_type_object = *object;
            }
            break;
        }
        case MemType::SHARED_BUF: {
            auto object = extract_object(remote_properties, mem_handle);
            if (object.has_value()) {
                _mem_handle_object = *object;
            }
            break;
        }
        default:
            OPENVINO_THROW("Unsupported shared object type ", *_mem_type);
        }
    }
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return _properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    if (!params.empty()) {
        _mem_type = extract_object(params, mem_type);

        switch (*_mem_type) {
        case MemType::L0_INTERNAL_BUF: {
            auto object = extract_object(params, tensor_type, false);
            if (object.has_value()) {
                _tensor_type_object = *object;
            }
            break;
        }
        case MemType::SHARED_BUF: {
            auto object = extract_object(params, mem_handle);
            if (object.has_value()) {
                _mem_handle_object = *object;
            }
            break;
        }
        default:
            OPENVINO_THROW("Unsupported shared object type ", *_mem_type);
        }
    }

    return _device->createRemoteTensor(get_this_shared_ptr(),
                                       type,
                                       shape,
                                       _config,
                                       _tensor_type_object,
                                       *_mem_type,
                                       _mem_handle_object);
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return _device->createHostTensor(get_this_shared_ptr(),
                                     type,
                                     shape,
                                     _config,
                                     _tensor_type_object,
                                     *_mem_type,
                                     _mem_handle_object);
}

const std::string& RemoteContextImpl::get_device_name() const {
    return _device_name;
}

std::shared_ptr<ov::IRemoteContext> RemoteContextImpl::get_this_shared_ptr() {
    return shared_from_this();
}

}  // namespace intel_npu
