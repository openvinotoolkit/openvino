// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <inference_engine.hpp>
#include "infer_request_wrap.hpp"

class RemoteContextHelper {
    class Impl;
    std::unique_ptr<Impl> _impl;
public:
    RemoteContextHelper();
    ~RemoteContextHelper();

    void Init(InferenceEngine::Core& ie);
    void PreallocRemoteMem(InferReqWrap::Ptr& request,
        const std::string& inputBlobName,
        const InferenceEngine::Blob::Ptr& inputBlob);
    InferenceEngine::RemoteContext::Ptr getRemoteContext();
};
