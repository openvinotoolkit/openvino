// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file containing function copying Blob's in different layouts
 * @file blob_transform.hpp
 */

#pragma once

#include "ie_api.h"
#include "ie_blob.h"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {

/**
 * @brief      Copies data with taking into account layout and precision params
 * @ingroup    ie_dev_api_memory
 *
 * @param[in]  src   The source Blob::Ptr
 * @param[in]  dst   The destination Blob::Ptr
 */
INFERENCE_ENGINE_API_CPP(void) blob_copy(Blob::Ptr src, Blob::Ptr dst);

}  // namespace InferenceEngine
IE_SUPPRESS_DEPRECATED_END
