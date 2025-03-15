// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

namespace ov {
namespace intel_cpu {

RemoteContextImpl::RemoteContextImpl(const std::string& device_name) : m_device_name(device_name) {}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    // TODO: should we check `params` are not empty params?
    return create_host_tensor(type, shape);
}

const std::string& RemoteContextImpl::get_device_name() const {
    return m_device_name;
}

}  // namespace intel_cpu
}  // namespace ov
