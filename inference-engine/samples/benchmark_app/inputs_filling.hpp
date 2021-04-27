// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <string>
#include <vector>

#include "infer_request_wrap.hpp"
#include "utils.hpp"

void fillBlobs(const std::vector<std::string>& inputFiles, const size_t& batchSize, benchmark_app::InputsInfo& app_inputs_info,
               std::vector<InferReqWrap::Ptr> requests);
