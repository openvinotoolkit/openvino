// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

// clang-format off
#include "inference_engine.hpp"

#include "infer_request_wrap.hpp"
#include "utils.hpp"
// clang-format on

std::map<std::string, std::vector<InferenceEngine::Blob::Ptr>> prepareCachedBlobs(
    std::map<std::string, std::vector<std::string>>& inputFiles,
    std::vector<benchmark_app::InputsInfo>& app_inputs_info);
