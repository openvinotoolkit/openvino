// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/remote_context.hpp"

#include "cpp_interfaces/interface/ie_iremote_context.hpp"
#include "ie_remote_blob.hpp"

#define REMOTE_CONTEXT_STATEMENT(...)                                   \
    if (_impl == nullptr)                                               \
        IE_THROW(NotAllocated) << "RemoteContext was not initialized."; \
    try {                                                               \
        __VA_ARGS__;                                                    \
    } catch (...) {                                                     \
        ::InferenceEngine::details::Rethrow();                          \
    }

namespace ov {
namespace runtime {

RemoteContext::RemoteContext(const ie::details::SharedObjectLoader& so, const ie::IRemoteContext::Ptr& impl)
    : _so(so),
      _impl(impl) {
    if (_impl == nullptr)
        IE_THROW() << "RemoteContext was not initialized.";
}

std::string RemoteContext::get_device_name() const {
    REMOTE_CONTEXT_STATEMENT(return _impl->getDeviceName());
}

std::shared_ptr<ie::RemoteBlob> RemoteContext::create_blob(const ie::TensorDesc& tensorDesc,
                                                           const ie::ParamMap& params) {
    REMOTE_CONTEXT_STATEMENT(return _impl->CreateBlob(tensorDesc, params));
}

ie::ParamMap RemoteContext::get_params() const {
    REMOTE_CONTEXT_STATEMENT(return _impl->getParams());
}

}  // namespace runtime
}  // namespace ov
