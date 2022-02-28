// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "async_infer_request.hpp"
#include <ie_icore.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>

namespace MultiDevicePlugin {
using namespace InferenceEngine;

explicit BaseInferRequest::BaseInferRequest(const InferenceEngine::SoIInferRequestInternal&  inferRequest)
    :_realInferRequest(inferRequest),
    _schedule(schedule)
    {

    }

void BaseInferRequest::InferImpl() {
    _schedule.SetInferRequest(this);
    assert(_realInferRequest != nullptr);
    SetBlobsToAnotherRequest(_realInferRequest);
    _realInferRequest->infer();
}

}  // namespace MultiDevicePlugin
