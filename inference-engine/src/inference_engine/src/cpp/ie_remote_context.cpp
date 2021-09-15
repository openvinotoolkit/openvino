// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exception>

#include "openvino/core/except.hpp"
#include "openvino/runtime/remote_context.hpp"

#include "ie_remote_blob.hpp"
#include "ie_remote_context.hpp"

#define OV_REMOTE_CONTEXT_STATEMENT(...)                                     \
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized."); \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception & ex) {                                    \
        throw ov::Exception(ex.what());                                      \
    }

namespace ov {
namespace runtime {

RemoteContext::RemoteContext(const std::shared_ptr<void>& so, const ie::RemoteContext::Ptr& impl)
    : _so(so),
      _impl(impl) {
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized.");
}

std::string RemoteContext::get_device_name() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getDeviceName());
}

std::shared_ptr<ie::RemoteBlob> RemoteContext::create_blob(const ie::TensorDesc& tensorDesc,
                                                           const ie::ParamMap& params) {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->CreateBlob(tensorDesc, params));
}

ie::ParamMap RemoteContext::get_params() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getParams());
}

}  // namespace runtime
}  // namespace ov
