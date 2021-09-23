// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_remote_context.hpp"

#include <exception>

#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/remote_context.hpp"

#define OV_REMOTE_CONTEXT_STATEMENT(...)                                     \
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized."); \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        throw ov::Exception(ex.what());                                      \
    } catch (...) {                                                          \
        OPENVINO_ASSERT(false, "Unexpected exception");                      \
    }

namespace ov {
namespace runtime {

RemoteContext::RemoteContext(const std::shared_ptr<void>& so, const ie::RemoteContext::Ptr& impl)
    : _so{so},
      _impl{impl} {
    OPENVINO_ASSERT(_impl != nullptr, "RemoteContext was not initialized.");
}

std::string RemoteContext::get_device_name() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getDeviceName());
}

RemoteTensor RemoteContext::create_tensor(const element::Type& element_type,
                                          const Shape& shape,
                                          const ie::ParamMap& params) {
    OV_REMOTE_CONTEXT_STATEMENT({
        return {_so,
                _impl->CreateBlob({ie::details::convertPrecision(element_type),
                                   shape,
                                   ie::TensorDesc::getLayoutByRank(shape.size())},
                                  params)};
    });
}

ie::ParamMap RemoteContext::get_params() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getParams());
}

}  // namespace runtime
}  // namespace ov
