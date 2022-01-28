// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_remote_context.hpp"

#include <memory>
#include <string>

#include "blob_factory.hpp"

namespace InferenceEngine {

MemoryBlob::Ptr RemoteContext::CreateHostBlob(const TensorDesc& tensorDesc) {
    auto blob = std::dynamic_pointer_cast<MemoryBlob>(make_blob_with_precision(tensorDesc));
    if (!blob)
        IE_THROW(NotAllocated) << "Failed to create host blob in remote context for " << getDeviceName() << " device";

    return blob;
}

}  // namespace InferenceEngine
