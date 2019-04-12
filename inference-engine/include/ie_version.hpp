// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides versioning information for the inference engine shared library
 * @file ie_version.hpp
 */
#pragma once

#include "ie_api.h"

namespace InferenceEngine {

/**
 * @struct Version
 * @brief  Represents version information that describes plugins and the inference engine runtime library
 */
#pragma pack(push, 1)
struct Version {
    /**
     * @brief An API version reflects the set of supported features
     */
    struct {
        int major;
        int minor;
    } apiVersion;
    /**
     * @brief A null terminated string with build number
     */
    const char * buildNumber;
    /**
     * @brief A null terminated description string
     */
    const char * description;
};
#pragma pack(pop)

/**
 * @brief Gets the current Inference Engine version
 * @return The current Inference Engine version
 */
INFERENCE_ENGINE_API(const Version*) GetInferenceEngineVersion() noexcept;

}  // namespace InferenceEngine
