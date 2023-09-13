// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_remote_context.hpp"

#include <memory>
#include <string>

#include "blob_factory.hpp"
#include "dev/converter_utils.hpp"
#include "dev/remote_context_wrapper.hpp"
#include "openvino/runtime/remote_context.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/plugin.hpp"
#endif

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START
MemoryBlob::Ptr RemoteContext::CreateHostBlob(const TensorDesc& tensorDesc) {
    auto blob = std::dynamic_pointer_cast<MemoryBlob>(make_blob_with_precision(tensorDesc));
    if (!blob)
        IE_THROW(NotAllocated) << "Failed to create host blob in remote context for " << getDeviceName() << " device";

    return blob;
}

const std::shared_ptr<InferenceEngine::RemoteContext> RemoteContext::GetHardwareContext() {
#ifdef PROXY_PLUGIN_ENABLED
    if (auto wrapper = dynamic_cast<ov::RemoteContextWrapper*>(this)) {
        auto ov_context = wrapper->get_context();
        auto hw_context = ov::proxy::get_hardware_context(ov_context);
        return ov::legacy_convert::convert_remote_context(hw_context._ptr);
    }
#endif
    return shared_from_this();
}

const std::shared_ptr<const InferenceEngine::RemoteContext> RemoteContext::GetHardwareContext() const {
#ifdef PROXY_PLUGIN_ENABLED
    if (auto wrapper = dynamic_cast<const ov::RemoteContextWrapper*>(this)) {
        auto ov_context = wrapper->get_context();
        auto hw_context = ov::proxy::get_hardware_context(ov_context);
        return ov::legacy_convert::convert_remote_context(hw_context._ptr);
    }
#endif
    return shared_from_this();
}
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
