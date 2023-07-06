// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <memory>

#include "openvino/runtime/iremote_context.hpp"

ov::autobatch_plugin::RemoteContext::RemoteContext(ov::RemoteContext&& ctx) : m_context(std::move(ctx)) {}

const std::string& ov::autobatch_plugin::RemoteContext::get_device_name() const {
    OPENVINO_NOT_IMPLEMENTED;
}

const ov::AnyMap& ov::autobatch_plugin::RemoteContext::get_property() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteTensor> ov::autobatch_plugin::RemoteContext::create_tensor(const ov::element::Type& type,
                                                                                      const ov::Shape& shape,
                                                                                      const ov::AnyMap& params) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ITensor> ov::autobatch_plugin::RemoteContext::create_host_tensor(const ov::element::Type type,
                                                                                     const ov::Shape& shape) {
    OPENVINO_NOT_IMPLEMENTED;
}

const std::shared_ptr<ov::IRemoteContext>& ov::autobatch_plugin::RemoteContext::get_hardware_context() {
    if (m_context) {
        return m_context._impl;
    } else
        OPENVINO_THROW("Get hardware context failed, the remotext is NULL!");
}