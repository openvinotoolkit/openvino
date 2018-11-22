// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_api.h"

namespace InferenceEngine {

/**
 * @brief Check if CPU is x86 with SSE4.2
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_sse42();

}  // namespace InferenceEngine
