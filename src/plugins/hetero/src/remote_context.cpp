// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include <memory>

#include "openvino/runtime/make_tensor.hpp"
#include "remote_tensor.hpp"

namespace ov {
namespace hetero {

RemoteContext::RemoteContext(std::map<std::string, ov::SoPtr<ov::IRemoteContext>> contexts)
    : m_contexts(std::move(contexts)) {
    if (m_contexts.empty()) {
        OPENVINO_ASSERT("HETERO RemoteContext must have at least one underlying context");
    }
}
const ov::AnyMap& RemoteContext::get_property() const {
    return m_contexts.begin()->second->get_property();
}

std::shared_ptr<RemoteContext> RemoteContext::get_this_shared_ptr() {
    return std::static_pointer_cast<RemoteContext>(shared_from_this());
}

ov::SoPtr<ov::IRemoteTensor> RemoteContext::create_tensor(const ov::element::Type& type,
                                                          const ov::Shape& shape,
                                                          const ov::AnyMap& params) {
    std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors;
    tensors.reserve(m_contexts.size());
    for (const auto& item : m_contexts) {
        tensors.emplace_back(item.second->create_tensor(type, shape, params));
    }
    auto remote_tensor_ptr = std::make_shared<ov::hetero::RemoteTensor>(get_this_shared_ptr(), tensors);
    return ov::SoPtr<ov::IRemoteTensor>(remote_tensor_ptr);
}

const std::string& RemoteContext::get_device_name() const {
    static const std::string name = "HETERO";
    return name;
}

}  // namespace hetero
}  // namespace ov
