// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>

#include "utils.hpp"
#include "infer_request_wrap.hpp"

#include "remotecontext_helper.hpp"

#ifdef USE_REMOTE_MEM
void fillBlobs(RemoteContextHelper& remoteContextHelper,
               const std::vector<std::string>& inputFiles,
               const size_t& batchSize,
               benchmark_app::InputsInfo& app_inputs_info,
               std::vector<InferReqWrap::Ptr> requests, bool preallocImage);
#else
void fillBlobs(const std::vector<std::string>& inputFiles,
               const size_t& batchSize,
               benchmark_app::InputsInfo& app_inputs_info,
               std::vector<InferReqWrap::Ptr> requests);
#endif
