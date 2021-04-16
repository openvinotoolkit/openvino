// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef USE_REMOTE_MEM
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <inference_engine.hpp>
#include "infer_request_wrap.hpp"
#include "WorkloadContext.h"
#include "RemoteMemory.h"

/// @brief Helper class for RemoteContext. Creates remote context and memory.
class RemoteContextHelper {
public:
    void init(InferenceEngine::Core& ie);
    void preallocRemoteMem(InferReqWrap::Ptr& request,
        const std::string& inputBlobName,
        const InferenceEngine::Blob::Ptr& inputBlob);
    InferenceEngine::RemoteContext::Ptr getRemoteContext();

private:
    HddlUnite::RemoteMemory::Ptr allocateRemoteMemory(const void* data, 
        const size_t& dataSize);

    WorkloadID _workloadId = -1;
    HddlUnite::WorkloadContext::Ptr _context;
    InferenceEngine::RemoteContext::Ptr _contextPtr;
    bool _init = false;
};

#endif
