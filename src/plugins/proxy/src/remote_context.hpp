// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace ov {
namespace proxy {

class RemoteContext : public ov::IRemoteContext {
public:
    RemoteContext(const ov::RemoteContext& ctx)
        : m_name(ctx.get_device_name()),
          m_property(ctx.get_params()),
          m_context(ctx) {}
    const std::string& get_device_name() const override {
        m_name = m_context.get_device_name();
        return m_name;
    }

    const ov::AnyMap& get_property() const override {
        m_property = m_context.get_params();
        return m_property;
    }

    std::shared_ptr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                                     const ov::Shape& shape,
                                                     const ov::AnyMap& params = {}) override {
        return m_context._impl->create_tensor(type, shape, params);
    }

    std::shared_ptr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override {
        return m_context._impl->create_host_tensor(type, shape);
    }

private:
    mutable std::string m_name;
    mutable ov::AnyMap m_property;
    ov::RemoteContext m_context;
};

}  // namespace proxy
}  // namespace ov
