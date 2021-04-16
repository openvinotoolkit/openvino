// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef USE_REMOTE_MEM
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <regex>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include "utils.hpp"
#include "remotecontext_helper.hpp"
#include "hddl2/hddl2_params.hpp"
#include "ie_compound_blob.h"

using namespace InferenceEngine;

void RemoteContextHelper::init(InferenceEngine::Core& ie) {
    _context = HddlUnite::createWorkloadContext();
    _context->setContext(_workloadId);
    auto ret = registerWorkloadContext(_context);
    if (ret != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << "registerWorkloadContext failed with " << ret;
    }

    // init context map and create context based on it
    ParamMap paramMap = { {HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), _workloadId} };
    _contextPtr = ie.CreateContext("VPUX", paramMap);
    _init = true;
}

HddlUnite::RemoteMemory::Ptr RemoteContextHelper::allocateRemoteMemory(const void* data, const size_t& dataSize) {
    auto remoteFrame = std::make_shared<HddlUnite::RemoteMemory>(*_context,
        HddlUnite::RemoteMemoryDesc(dataSize, 1, dataSize, 1));

    if (remoteFrame == nullptr) {
        THROW_IE_EXCEPTION << "Failed to allocate remote memory.";
    }

    if (remoteFrame->syncToDevice(data, dataSize) != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to sync memory to device.";
    }
    return remoteFrame;
}

void RemoteContextHelper::preallocRemoteMem(InferReqWrap::Ptr& request,
    const std::string& inputBlobName,
    const Blob::Ptr& inputBlob) {
    if (_init == false)
        THROW_IE_EXCEPTION << "RemoteContextHelper did not init.";
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    const TensorDesc& inputTensor = minput->getTensorDesc();
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->rmap();
    auto inputBlobData = minputHolder.as<uint8_t*>();

    // 1, allocate memory with HddlUnite on device
    auto remoteMemory = allocateRemoteMemory(inputBlobData, minput->byteSize());

    // 2, create remote blob by using already exists remote memory and specify color format of it
    ParamMap blobParamMap = { {HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory} };
    RemoteBlob::Ptr remoteBlobPtr = _contextPtr->CreateBlob(inputTensor, blobParamMap);
    if (remoteBlobPtr == nullptr) {
        THROW_IE_EXCEPTION << "CreateBlob failed.";
    }

    // 3, set remote blob
    request->setBlob(inputBlobName, remoteBlobPtr);
}

RemoteContext::Ptr RemoteContextHelper::getRemoteContext() {
    if (_init == false)
        THROW_IE_EXCEPTION << "RemoteContextHelper did not init.";
    return _contextPtr;
}
#endif
