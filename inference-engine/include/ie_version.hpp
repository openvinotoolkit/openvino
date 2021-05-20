// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides versioning information for the Inference Engine library
 *
 * @file ie_version.hpp
 */
#pragma once

/**
 * @def IE_VERSION_MAJOR
 * @brief Defines Inference Engine major version
 *
 * @def IE_VERSION_MINOR
 * @brief Defines Inference Engine minor version
 *
 * @def IE_VERSION_PATCH
 * @brief Defines Inference Engine patch version
 */

#define IE_VERSION_MAJOR 2021
#define IE_VERSION_MINOR 4
#define IE_VERSION_PATCH 0

#include "ie_api.h"

/**
 * @brief Inference Engine C++ API
 */
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
        int major; //!< A major version
        int minor; //!< A minor version
    } apiVersion;
    /**
     * @brief A null terminated string with build number
     */
    const char* buildNumber;
    /**
     * @brief A null terminated description string
     */
    const char* description;
};
#pragma pack(pop)

/**
 * @brief Gets the current Inference Engine version
 *
 * @return The current Inference Engine version
 */
INFERENCE_ENGINE_API(const Version*) GetInferenceEngineVersion() noexcept;

}  // namespace InferenceEngine
