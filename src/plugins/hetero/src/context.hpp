// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iremote_context.hpp"

#include <string>
// #include <map>

namespace ov {
namespace hetero {
class HeteroContext : public ov::IRemoteContext {
public:
    using Ptr = std::shared_ptr<HeteroContext>;

    HeteroContext(std::map<std::string, ov::SoPtr<ov::IRemoteContext>> contexts);

    const std::string& get_device_name() const override;
    const ov::AnyMap& get_property() const override;

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params) override;

private:
    std::shared_ptr<HeteroContext> get_this_shared_ptr();
    std::map<std::string, ov::SoPtr<ov::IRemoteContext>> m_contexts;
};

}  // namespace hetero
}  // namespace ov