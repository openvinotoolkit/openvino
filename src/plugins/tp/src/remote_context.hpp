// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace tp {

/// The context of a TP compiled model. A TP model has no single device, so it
/// gives out host memory only: allocated through rank 0's context, which every
/// rank device can read.
class RemoteContext : public ov::IRemoteContext {
public:
    RemoteContext(std::vector<std::string> device_names, ov::SoPtr<ov::IRemoteContext> rank0_context);

    const std::string& get_device_name() const override;

    /// The rank devices, as `ov::execution_devices`.
    const ov::AnyMap& get_property() const override;

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params) override;

    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;

private:
    std::string m_device_name{"TP"};
    ov::AnyMap m_properties;
    ov::SoPtr<ov::IRemoteContext> m_rank0_context;
};

}  // namespace tp
}  // namespace ov
