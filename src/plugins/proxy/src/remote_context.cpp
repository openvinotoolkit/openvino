// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

ov::proxy::RemoteContext::RemoteContext(const ov::RemoteContext& ctx)
    : m_name(ctx.get_device_name()),
      m_property(ctx.get_params()),
      m_context(ctx) {}

const std::string& ov::proxy::RemoteContext::get_device_name() const {
    m_name = m_context.get_device_name();
    return m_name;
}

const ov::AnyMap& ov::proxy::RemoteContext::get_property() const {
    m_property = m_context.get_params();
    return m_property;
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
