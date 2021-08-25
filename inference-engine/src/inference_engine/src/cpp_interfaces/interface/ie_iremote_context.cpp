// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp_interfaces/interface/ie_iremote_context.hpp>

namespace InferenceEngine {
std::string IRemoteContext::getDeviceName() const noexcept {
    return {};
}

RemoteBlob::Ptr IRemoteContext::CreateBlob(const TensorDesc&, const ParamMap&) {
    IE_THROW(NotImplemented);
}

ParamMap IRemoteContext::getParams() const {
    IE_THROW(NotImplemented);
}
}  // namespace InferenceEngine
