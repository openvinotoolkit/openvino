// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp {

RemoteContext::RemoteContext(std::vector<std::string> device_names, ov::SoPtr<ov::IRemoteContext> rank0_context)
    : m_properties{{ov::execution_devices.name(), std::move(device_names)}},
      m_rank0_context(std::move(rank0_context)) {
    OPENVINO_ASSERT(m_rank0_context, "[TP] the context of rank 0 is missing");
}

const std::string& RemoteContext::get_device_name() const {
    return m_device_name;
}

const ov::AnyMap& RemoteContext::get_property() const {
    return m_properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContext::create_tensor(const ov::element::Type&,
                                                          const ov::Shape&,
                                                          const ov::AnyMap&) {
    // A device tensor would be one model-level tensor on one rank, while every
    // rank expects its own sharded one.
    OPENVINO_THROW("TP: a TP model has no single device; create_tensor() is not supported (use create_host_tensor()).");
}

ov::SoPtr<ov::ITensor> RemoteContext::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return m_rank0_context->create_host_tensor(type, shape);
}

}  // namespace tp
}  // namespace ov
