// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace ov {
namespace autobatch_plugin {

class RemoteContext : public ov::IRemoteContext {
public:
    explicit RemoteContext(ov::RemoteContext&& ctx);

    const std::string& get_device_name() const override;

    const ov::AnyMap& get_property() const override;

    std::shared_ptr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                                     const ov::Shape& shape,
                                                     const ov::AnyMap& params = {}) override;

    std::shared_ptr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;

    const std::shared_ptr<ov::IRemoteContext>& get_hardware_context();

private:
    ov::RemoteContext m_context;
};

}  // namespace autobatch_plugin
}  // namespace ov