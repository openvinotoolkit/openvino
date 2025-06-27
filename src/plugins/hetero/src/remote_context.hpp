// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace hetero {
class RemoteContext : public ov::IRemoteContext {
public:
    using Ptr = std::shared_ptr<RemoteContext>;

    RemoteContext(std::map<std::string, ov::SoPtr<ov::IRemoteContext>> contexts);

    const std::string& get_device_name() const override;
    const ov::AnyMap& get_property() const override;

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params) override;

private:
    std::shared_ptr<RemoteContext> get_this_shared_ptr();
    std::map<std::string, ov::SoPtr<ov::IRemoteContext>> m_contexts;
};

}  // namespace hetero
}  // namespace ov