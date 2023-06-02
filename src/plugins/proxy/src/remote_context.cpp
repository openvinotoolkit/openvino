// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <memory>

#include "proxy_plugin.hpp"

ov::proxy::RemoteContext::RemoteContext(const ov::RemoteContext& ctx, const std::string& dev_name)
    : m_name(dev_name),
      m_context(ctx) {}

const std::string& ov::proxy::RemoteContext::get_device_name() const {
    return m_name;
}

const ov::AnyMap& ov::proxy::RemoteContext::get_property() const {
    return m_context._impl->get_property();
}

std::shared_ptr<ov::IRemoteTensor> ov::proxy::RemoteContext::create_tensor(const ov::element::Type& type,
                                                                           const ov::Shape& shape,
                                                                           const ov::AnyMap& params) {
    return m_context._impl->create_tensor(type, shape, params);
}

std::shared_ptr<ov::ITensor> ov::proxy::RemoteContext::create_host_tensor(const ov::element::Type type,
                                                                          const ov::Shape& shape) {
    return m_context._impl->create_host_tensor(type, shape);
}

const ov::RemoteContext& ov::proxy::RemoteContext::get_hardware_context(const ov::RemoteContext& context) {
    if (auto proxy_context = std::dynamic_pointer_cast<ov::proxy::RemoteContext>(context._impl)) {
        return proxy_context->m_context;
    }
    return context;
}

const ov::RemoteContext& ov::proxy::get_hardware_context(const ov::RemoteContext& context) {
    return ov::proxy::RemoteContext::get_hardware_context(context);
}
