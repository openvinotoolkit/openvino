// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "remote_tensor.hpp"

ov::autobatch_plugin::RemoteContext::RemoteContext(ov::RemoteContext&& ctx, const std::string& dev_name)
    : m_context(std::move(ctx)),
      m_name(dev_name) {}

const std::string& ov::autobatch_plugin::RemoteContext::get_device_name() const {
    return m_name;
}

const ov::AnyMap& ov::autobatch_plugin::RemoteContext::get_property() const {
    return m_context._impl->get_property();
}

std::shared_ptr<ov::IRemoteTensor> ov::autobatch_plugin::RemoteContext::create_tensor(const ov::element::Type& type,
                                                                                      const ov::Shape& shape,
                                                                                      const ov::AnyMap& params) {
    return std::make_shared<ov::autobatch_plugin::RemoteTensor>(m_context.create_tensor(type, shape, params), m_name);
}

std::shared_ptr<ov::ITensor> ov::autobatch_plugin::RemoteContext::create_host_tensor(const ov::element::Type type,
                                                                                     const ov::Shape& shape) {
    return m_context._impl->create_host_tensor(type, shape);
}