// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace runtime {
ie::ParamMap RemoteTensor::get_params() const {
    OPENVINO_ASSERT(_impl != nullptr, "Remote tensor was not initialized.");
    auto remote_impl = InferenceEngine::as<InferenceEngine::RemoteBlob>(_impl);
    OPENVINO_ASSERT(remote_impl != nullptr, "Remote tensor was not initialized using remote implementation");
    try {
        return remote_impl->getParams();
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

std::string RemoteTensor::get_device_name() const {
    OPENVINO_ASSERT(_impl != nullptr, "Remote tensor was not initialized.");
    auto remote_impl = InferenceEngine::as<InferenceEngine::RemoteBlob>(_impl);
    OPENVINO_ASSERT(remote_impl != nullptr, "Remote tensor was not initialized using remote implementation");
    try {
        return remote_impl->getDeviceName();
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}
}  // namespace runtime
}  // namespace ov
