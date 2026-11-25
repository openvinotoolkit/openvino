// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/remote_context.hpp"

namespace ov::intel_gpu {

RemoteContextImpl::RemoteContextImpl(const std::map<std::string, RemoteContextImpl::Ptr>& /*known_contexts*/, const AnyMap& /*params*/) {
    OPENVINO_THROW("[GPU][Vulkan] External GPU contexts are unsupported");
}

void RemoteContextImpl::init_default_context_type() {}

void RemoteContextImpl::init_properties() {}

// Interop implementations use the engine and handles of the owning context.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor_from_params(const ov::element::Type& /*type*/,
                                                                          const ov::Shape& /*shape*/,
                                                                          const ov::AnyMap& /*params*/) {
    OPENVINO_THROW("[GPU][Vulkan] External GPU tensors are unsupported");
}

}  // namespace ov::intel_gpu
