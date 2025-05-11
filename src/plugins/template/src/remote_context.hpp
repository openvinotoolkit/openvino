// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace template_plugin {

// ! [remote_context:header]
class RemoteContext : public ov::IRemoteContext {
public:
    RemoteContext();
    const std::string& get_device_name() const override;
    const ov::AnyMap& get_property() const override;
    ov::SoPtr<IRemoteTensor> create_tensor(const ov::element::Type& type,
                                           const ov::Shape& shape,
                                           const ov::AnyMap& params = {}) override;

private:
    std::string m_name;
    ov::AnyMap m_property;
};
// ! [remote_context:header]

}  // namespace template_plugin
}  // namespace ov
