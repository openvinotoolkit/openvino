// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_api.h"
#include "ie_blob.h"

namespace InferenceEngine {

/**
 * @brief Data copy with taking into account layout and precision params
 */
INFERENCE_ENGINE_API_CPP(void) blob_copy(Blob::Ptr src, Blob::Ptr dst);

}  // namespace InferenceEngine
